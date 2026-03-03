#!/usr/bin/env bash
set -euo pipefail

# huggingface にログイン
# ./.venv/bin/huggingface-cli login --token "$HF_TOKEN"
# ./.venv/bin/huggingface-cli whoami

# 保存先を指定して，huggingface のモデルをダウンロード
# DEST=./checkpoints/teacher/pi05_libero
# mkdir -p "$DEST"
# hf download bf-jeon/pi05_libero --local-dir "$DEST"


# JAX ModelをPyTorchに変換するコマンド
# uv run examples/convert_jax_model_to_pytorch.py \
#     --checkpoint_dir ./checkpoints/teacher/pi05_libero \
#     --config_name pi05_libero \
#     --output_path ./checkpoints/teacher/pi05_libero/pytorch

# ======================
# 教師モデルの学習コマンド
# ======================

PEAK_LRS=(4e-5)
# PEAK_LRS=(2.5e-5 5e-5 1e-4)
SEED=42
EXP_BASE="finetune_libero"

# pi0_liberoの学習コマンド
# python scripts/compute_norm_stats.py --config-name=pi0_libero
# python -m torch.distributed.run \
#   --standalone --nnodes=1 --nproc_per_node=4 \
#   scripts/train_pytorch.py pi0_libero --exp_name fineturne_libero

# pi05_liberoの学習コマンド
# python scripts/compute_norm_stats.py --config-name=pi05_libero
# ./.venv/bin/python -m torch.distributed.run \
#   --standalone --nnodes=1 --nproc_per_node=4 \
#   scripts/train_pytorch.py pi05_libero --exp_name fineturne_libero --seed 42

# PEAK_LRSの値を変えて複数回学習を実行するコマンド
for peak_lr in "${PEAK_LRS[@]}"; do
  # decay_lr = peak_lr / 10 を計算
  decay_lr=$(./.venv/bin/python - <<PY
lr=float("${peak_lr}")
print(lr/10.0)
PY
)

  exp_name="${EXP_BASE}_${peak_lr}"

  ./.venv/bin/python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "${exp_name}" \
    --seed "${SEED}" \
    --lr_schedule.peak_lr "${peak_lr}" \
    --lr_schedule.decay_lr "${decay_lr}"
done


# ======================
# 生徒モデルの学習コマンド
# ======================

# 6層のモデルに蒸留
# .venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_libero_distill_6
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
# ./.venv/bin/python -m torch.distributed.run \
#   --standalone --nnodes=1 --nproc_per_node=4 \
#   scripts/distill_pytorch.py pi0_libero_distill_6 \
#   --exp_name only_teacher \
#   --seed 43


# ======================
# 評価コマンド
# ======================

# LIBERO評価実行コマンド（デフォルト）
# MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python examples/libero/main.py
# uv run scripts/serve_policy.py --env LIBERO --train_seed 43

# LIBERO評価実行コマンド（モデルと保存場所指定）
# export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
# MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 ./examples/libero/.venv/bin/python examples/libero/main.py \
#   --args.result_root ./result/libero/student/pi0_libero_distill_6/libero_distill_6/seed43/30000
# ./.venv/bin/python scripts/serve_policy.py \
#   --env LIBERO \
#   --port 8000 \
#   policy:checkpoint \
#   --policy.config pi0_libero_distill_6 \
#   --policy.dir ./checkpoints/student/pi0_libero_distill_6/libero_distill_6/seed43/30000
