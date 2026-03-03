"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""
from __future__ import annotations

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import matplotlib.pyplot as plt
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


# =========================
# ユーティリティ関数
# =========================
def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device

def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)

def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()

def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )

def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )

def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)

def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise

def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None

def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB," 
        f"reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB,"
        f"peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )

def freeze_module(m: torch.nn.Module) -> torch.nn.Module:
    m.eval()
    if hasattr(m, "requires_grad_"):
        m.requires_grad_(False)
    else:
        for p in m.parameters():
            p.requires_grad_(False)
    return m

def _format_num_mb(n: int) -> str:
    """123 -> '123', 12_300 -> '12.3K', 12_300_000 -> '12.3M', 1_230_000_000 -> '1.23B'"""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)

def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    """(total_params, trainable_params)"""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable

def log_student_teacher_params(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    *,
    is_main: bool = True,
    logger: logging.Logger | None = None,
    title: str = "モデルのパラメータ数",
):
    if not is_main:
        return
    if logger is None:
        logger = logging.getLogger(__name__)

    s_total, s_train = _count_params(student_model)
    t_total, t_train = _count_params(teacher_model)

    s_ratio = (s_train / s_total * 100.0) if s_total > 0 else 0.0
    t_ratio = (t_train / t_total * 100.0) if t_total > 0 else 0.0

    logger.info("============================================================")
    logger.info(f"{title}")
    logger.info("------------------------------------------------------------")
    logger.info(
        f"生徒モデル: 総 { _format_num_mb(s_total) } ({s_total:,}) | "
        f"学習 { _format_num_mb(s_train) } ({s_train:,}) | 学習率 {s_ratio:.2f}%"
    )
    logger.info(
        f"教師モデル: 総 { _format_num_mb(t_total) } ({t_total:,}) | "
        f"学習 { _format_num_mb(t_train) } ({t_train:,}) | 学習率 {t_ratio:.2f}%"
    )
    logger.info("============================================================")
    
@dataclass
class LossIntervalTracker:
    """
    log_interval ごとに loss / loss_gt / loss_teacher の平均を計算して保存する。
    さらに、平均推移を横3図(1行3列)で保存できる。

    想定:
      - update(): 各stepで生損失を足し込む
      - flush(): log_interval のタイミングで平均を確定し、履歴に保存
      - save(): 履歴を図として保存
    """
    log_interval: int
    out_path: Path

    # 生の蓄積（interval中）
    _sum_loss: float = 0.0
    _sum_loss_gt: float = 0.0
    _sum_loss_teacher: float = 0.0
    _n: int = 0

    # 平均履歴
    steps: List[int] = field(default_factory=list)
    loss_mean: List[float] = field(default_factory=list)
    loss_gt_mean: List[float] = field(default_factory=list)
    loss_teacher_mean: List[float] = field(default_factory=list)

    def update(self, *, loss: float, loss_gt: float, loss_teacher: float) -> None:
        """各stepの損失(スカラー)を加算する。"""
        self._sum_loss += float(loss)
        self._sum_loss_gt += float(loss_gt)
        self._sum_loss_teacher += float(loss_teacher)
        self._n += 1

    def flush(self, *, global_step: int) -> Optional[Dict[str, float]]:
        """
        log_interval 到達時に平均を確定して履歴に格納。
        それ以外は None を返す。
        """
        if self._n <= 0:
            return None

        # 「global_step が log_interval 境界のときだけ確定」したい場合
        # 呼び出し側で (global_step % log_interval == 0) の時だけ flush() するなら、
        # ここで判定は不要。安全のため入れるなら↓
        if global_step % self.log_interval != 0:
            return None

        mean_loss = self._sum_loss / self._n
        mean_loss_gt = self._sum_loss_gt / self._n
        mean_loss_teacher = self._sum_loss_teacher / self._n

        self.steps.append(int(global_step))
        self.loss_mean.append(mean_loss)
        self.loss_gt_mean.append(mean_loss_gt)
        self.loss_teacher_mean.append(mean_loss_teacher)

        # intervalをリセット
        self._sum_loss = 0.0
        self._sum_loss_gt = 0.0
        self._sum_loss_teacher = 0.0
        self._n = 0

        return {
            "loss": mean_loss,
            "loss_gt": mean_loss_gt,
            "loss_teacher": mean_loss_teacher,
        }

    def save(self, *, title: str = "Loss curves (interval mean)", dpi: int = 160) -> Path:
        """
        横3図で保存する。色指定はしない（デフォルトカラー）。
        """
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(title)

        x = np.asarray(self.steps, dtype=np.int64)

        # 1) loss
        axes[0].plot(x, self.loss_mean)
        axes[0].set_title("loss (mean per interval)")
        axes[0].set_xlabel("global_step")
        axes[0].set_ylabel("loss")
        axes[0].grid(True)

        # 2) loss_gt
        axes[1].plot(x, self.loss_gt_mean)
        axes[1].set_title("loss_gt (mean per interval)")
        axes[1].set_xlabel("global_step")
        axes[1].set_ylabel("loss_gt")
        axes[1].grid(True)

        # 3) loss_teacher
        axes[2].plot(x, self.loss_teacher_mean)
        axes[2].set_title("loss_teacher (mean per interval)")
        axes[2].set_xlabel("global_step")
        axes[2].set_ylabel("loss_teacher")
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(self.out_path, dpi=dpi)
        plt.close(fig)
        return self.out_path

def make_loss_plot_path(config, exp_checkpoint_dir: Path) -> Path:
    return "student"/ exp_checkpoint_dir / "loss_curves.png"


# =========================
# メイン学習ループ
# ========================
def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # 保存する/復元するチェックポイントディレクトリの設定
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # チェックポイントディレクトリの作成（存在しない場合のみ）
    if not resuming:
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # wandb 初期化
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # world size とバッチサイズの設定
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # データローダーの構築
    loader, data_config = build_datasets(config)

    # サンプルバッチの可視化とメモリ解放
    if is_main and config.wandb_enabled and not resuming:
        # データローダーからサンプルバッチを取得して可視化する
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # カメラビューを横に連結してwandbにログ出力する
        images_to_log = []
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # サンプルバッチとデータローダーをメモリから解放する
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # モデル構築を設定
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    # 教師モデルを構築（重みは後で読み込む）
    model = openpi.models_pytorch.pi0_pytorch.DistilledPI0Pytorch(model_cfg).to(device)
    assert hasattr(model_cfg, "teacher_config")
    assert hasattr(config, "pytorch_weight_path_teacher")
    teacher_config = _config.get_config(model_cfg.teacher_config).model
    object.__setattr__(teacher_config, "dtype", config.pytorch_training_precision)
    teacher_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(teacher_config).to(device)
    teacher_model.eval()

    # 教師モデルをファインチューニングした重みで初期化
    teacher_model_weight_path = config.pytorch_weight_path if \
        config.pytorch_weight_path_teacher is None \
            else config.pytorch_weight_path_teacher
    teacher_model_path = os.path.join(teacher_model_weight_path, "model.safetensors")
    safetensors.torch.load_model(
        teacher_model, teacher_model_path, strict=False
    )
    logging.info(f"教師モデルの重みを読み込みました: {teacher_model_path}")
    teacher_model = teacher_model.eval()
    teacher_model = freeze_module(teacher_model)
    assert not any(p.requires_grad for p in teacher_model.parameters())

    # gradient checkpointing をサポートしている場合は有効化（大規模モデルのメモリ最適化に有効）
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("メモリ最適化のために gradient checkpointing を有効化しました")
    else:
        enable_gradient_checkpointing = False
        logging.info("このモデルでは gradient checkpointing はサポートされていません")

    # モデル構築後のGPUメモリ使用量をログ出力
    if is_main and torch.cuda.is_available():
        logging.info("モデル構築後の初期GPUメモリ使用量をログ出力")
        log_memory_usage(device, 0, "after_model_creation")

    # DDPの設定（モデルのラッピング、メモリ最適化の有効化など）
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )

    # 生徒モデルを事前学習済み重みで初期化
    if config.pytorch_weight_path is not None:
        logging.info(f"重みを読み込み中: {config.pytorch_weight_path}")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors_path=config.pytorch_weight_path
        state_dict = safetensors.torch.load_file(os.path.join(str(safetensors_path), "model.safetensors"), device="cpu")
        model_dict = model_to_load.state_dict()
        # 形状が一致するものだけをフィルタリング
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                skipped_keys.append(k)
        # レイヤーのサブサンプリングによるマッピング
        if teacher_model_weight_path == config.pytorch_weight_path:
            l_teacher = teacher_config.gemma_depth
            l_student = model_cfg.gemma_depth
            layer_mapping = torch.linspace(0, l_teacher - 1, steps=l_student).round().long().tolist()
            mapping_target_list = [
                "paligemma_with_expert.gemma_expert.model.layers.",
                "paligemma_with_expert.paligemma.model.language_model.layers.",
            ]
            for s_idx, t_idx in enumerate(layer_mapping):
                for k in filtered_state_dict:
                    if k.startswith((f"{mapping_target_list[0]}{s_idx}"), k.startswith(f"{mapping_target_list[1]}{s_idx}")):
                        filtered_state_dict[k] = state_dict[k.replace(str(s_idx), str(t_idx))]
                        print(f"mapping teacher {k.replace(str(s_idx), str(t_idx))} -> student {k}")
        missing, unexpected = model_to_load.load_state_dict(filtered_state_dict, strict=False)

    # 設定から学習率スケジュールのパラメータを取得
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # 最適化関数の構築
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # 再開の場合はチェックポイントからモデルとオプティマイザの状態を復元
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")
    
    # 学習率スケジュール関数の定義（ウォームアップ + コサイン減衰）
    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # 学習に使用する変数を設定
    model.train()
    start_time = time.time()
    infos = []
    delta_loss = None
    delta_gt = None
    delta_teacher = None
    if "prev_means" not in locals():
        prev_means = None
    
    # 学習前の設定ログ出力
    if is_main:
        student_to_count = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        log_student_teacher_params(
            student_to_count,
            teacher_model,
            is_main=is_main,
            logger=logging.getLogger(__name__),
            title="蒸留(Shallow-π) 学習開始前のパラメータ数",
        )
        logging.info(
            f"学習開始: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"学習設定: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"メモリ最適化: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"学習率スケジュール: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"最適化関数: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # 進捗可視化設定
    loss_plot_path = make_loss_plot_path(config, exp_checkpoint_dir)
    loss_tracker = LossIntervalTracker(log_interval=config.log_interval, out_path=loss_plot_path)
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    # -------------
    # 学習ループ開始
    # -------------
    while global_step < config.num_train_steps:
        # 分散学習用にエポックを設定
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # 学習ステップが最大に達したら終了
            if global_step >= config.num_train_steps:
                break

            # 統一されたデータローダーから (observation, actions) を取得
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32)
            actions = actions.to(device)

            # 学習率を更新
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # 順伝播
            loss, loss_gt, loss_teacher = model(
                observation,
                actions,
                teacher=teacher_model,
            )

            # 逆伝播
            loss.backward()

            # メモリの使用量をログ出力（最初の数ステップのみ）
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # 勾配クリッピング
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # パラメータ更新
            optim.step()
            optim.zero_grad(set_to_none=True)

            # 勾配をより積極的にクリア
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # 統計を収集
            if is_main:
                loss_tracker.update(
                    loss=float(loss.item()),
                    loss_gt=float(loss_gt.item()),
                    loss_teacher=float(loss_teacher.item()),
                )
                infos.append(
                    {
                        "loss": float(loss.item()),
                        "loss_gt": float(loss_gt.item()),
                        "loss_teacher": float(loss_teacher.item()),
                        "learning_rate": float(optim.param_groups[0]["lr"]),
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    }
                )

            # ログ出力と図を更新
            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                
                # interval平均
                avg_loss_gt = sum(info["loss_gt"] for info in infos) / len(infos)
                avg_loss_teacher = sum(info["loss_teacher"] for info in infos) / len(infos)

                # “円滑さ”の簡易指標
                teacher_ratio = avg_loss_teacher / max(avg_loss, 1e-12)

                if prev_means is not None:
                    delta_loss = avg_loss - prev_means["loss"]
                    delta_gt = avg_loss_gt - prev_means["loss_gt"]
                    delta_teacher = avg_loss_teacher - prev_means["loss_teacher"]

                # “円滑さ”を判定
                status = "OK"
                if prev_means is not None:
                    if (delta_loss is not None) and (delta_loss > 0) and (delta_gt is not None) and (delta_gt > 0):
                        status = "WARN(total↑ & gt↑)"
                    elif (delta_loss is not None) and (delta_loss > 0) and (delta_teacher is not None) and (delta_teacher > 0) and (delta_gt is not None) and (delta_gt < 0):
                        status = "WARN(teacher↑ while gt↓) 競合/不整合の疑い"
                    elif (delta_loss is not None) and (abs(delta_loss) < 1e-4):
                        status = "STALL"

                msg = (
                    f"[smooth={status}] step={global_step} "
                    f"loss={avg_loss:.4f} (Δ{delta_loss:+.4f}) " if prev_means else f"[smooth={status}] step={global_step} loss={avg_loss:.4f} "
                )
                if prev_means:
                    msg = (
                        f"[smooth={status}] step={global_step} "
                        f"loss={avg_loss:.4f}(Δ{delta_loss:+.4f}) "
                        f"gt={avg_loss_gt:.4f}(Δ{delta_gt:+.4f}) "
                        f"teacher={avg_loss_teacher:.4f}(Δ{delta_teacher:+.4f}) "
                        f"t_ratio={teacher_ratio:.2f} "
                        f"lr={avg_lr:.2e} "
                        + (f"grad_norm={avg_grad_norm:.2f} " if avg_grad_norm is not None else "")
                        + f"time={elapsed:.1f}s"
                    )
                else:
                    msg = (
                        f"[smooth={status}] step={global_step} "
                        f"loss={avg_loss:.4f} gt={avg_loss_gt:.4f} teacher={avg_loss_teacher:.4f} "
                        f"t_ratio={teacher_ratio:.2f} lr={avg_lr:.2e} "
                        + (f"grad_norm={avg_grad_norm:.2f} " if avg_grad_norm is not None else "")
                        + f"time={elapsed:.1f}s"
                    )

                logging.info(msg)
                prev_means = {"loss": avg_loss, "loss_gt": avg_loss_gt, "loss_teacher": avg_loss_teacher}

                # wandb にログを記録
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []
                means = loss_tracker.flush(global_step=global_step)

                # 図を逐次更新して保存
                if means is not None:
                    loss_tracker.save(title="Loss curves")

            # config.save_interval でチェックポイントを保存
            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # プログレスバーを更新
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    if pbar is not None:
        pbar.close()

    if is_main and config.wandb_enabled:
        wandb.finish()
        
    if is_main:
        loss_tracker.save(title="Loss curves (final)")
        logging.info(f"損失を可視化: {loss_plot_path}")

    cleanup_ddp()


# ========================
# エントリーポイント
# ========================
def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
