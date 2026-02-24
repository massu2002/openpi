# 保存先を指定して，huggingface のモデルをダウンロード
# DEST=./checkpoints/teacher/pi05_libero
# mkdir -p "$DEST"
# hf download bf-jeon/pi05_libero --local-dir "$DEST"


# JAX ModelをPyTorchに変換するコマンド
# uv run examples/convert_jax_model_to_pytorch.py \
#     --checkpoint_dir ./checkpoints/teacher/pi05_libero \
#     --config_name pi05_libero \
#     --output_path ./checkpoints/teacher/pi05_libero/pytorch


# 教師モデルの学習コマンド

# pi0_liberoの学習コマンド
# python scripts/compute_norm_stats.py --config-name=pi0_libero
# python -m torch.distributed.run \
#   --standalone --nnodes=1 --nproc_per_node=4 \
#   scripts/train_pytorch.py pi0_libero --exp_name fineturne_libero

# pi05_liberoの学習コマンド
# python scripts/compute_norm_stats.py --config-name=pi05_libero
python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_pytorch.py pi05_libero --exp_name fineturne_libero --seed 43

# 生徒モデルの学習コマンド

# 6層のモデルに蒸留
# ./.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_libero_distill_6

# ./.venv/bin/python -m torch.distributed.run \
#   --standalone --nnodes=1 --nproc_per_node=4 \
#   scripts/distill_pytorch.py pi0_libero_distill_6 --exp_name libero_distill_6


# LIBERO評価実行コマンド
# MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python examples/libero/main.py
# python scripts/serve_policy.py --env LIBERO
