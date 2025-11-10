#!/usr/bin/env bash
set -euo pipefail

# 在此处直接设置训练超参数
EPOCHS=10
BATCH_SIZE=128
MAX_LEN=128
D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_LAYERS=6
DROPOUT=0.3
LR=5e-4
WARMUP_STEPS=4000
WEIGHT_DECAY=1e-4
LABEL_SMOOTHING=0.1
GRAD_CLIP=1.0

# 可选：指定 GPU
# export CUDA_VISIBLE_DEVICES=0

PY=../src/train.py

python3 "$PY" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_len "$MAX_LEN" \
  --d_model "$D_MODEL" \
  --num_heads "$NUM_HEADS" \
  --d_ff "$D_FF" \
  --num_layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --warmup_steps "$WARMUP_STEPS" \
  --weight_decay "$WEIGHT_DECAY" \
  --label_smoothing "$LABEL_SMOOTHING" \
  --grad_clip "$GRAD_CLIP"