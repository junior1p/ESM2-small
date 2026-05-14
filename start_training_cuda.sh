#!/bin/bash
# ESM-2 style protein language model training on NVIDIA GPU (CUDA + BF16 AMP)
# Usage: bash start_training_cuda.sh [model_size]
#   model_size: 8M (default), 35M, 150M

set -e

MODEL_SIZE="${1:-8M}"
DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./output/${MODEL_SIZE}}"

echo "=== ESM-2 Training on CUDA (BF16 AMP) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Data dir   : ${DATA_DIR}"
echo "Output dir : ${OUT_DIR}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "==========================================="

BATCH_SIZE=32
GRAD_ACCUM=1
LR=1e-4
WARMUP=1000
EMA_DECAY=0.0

if [ "${MODEL_SIZE}" = "35M" ]; then
    BATCH_SIZE=16; GRAD_ACCUM=2; LR=5e-5; WARMUP=2000; EMA_DECAY=0.999
elif [ "${MODEL_SIZE}" = "150M" ]; then
    BATCH_SIZE=8; GRAD_ACCUM=4; LR=3e-5; WARMUP=4000; EMA_DECAY=0.999
fi

exec python3 train.py \
    --train_data "${DATA_DIR}/swissprot_train.fasta" \
    --val_data   "${DATA_DIR}/swissprot_val.fasta" \
    --model_size "${MODEL_SIZE}" \
    --max_len    512 \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr         "${LR}" \
    --warmup_steps "${WARMUP}" \
    --epochs     5 \
    --device     cuda \
    --ema_decay  "${EMA_DECAY}" \
    --out_dir    "${OUT_DIR}" \
    --seed       42
