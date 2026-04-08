#!/bin/bash
# Start MLU370 protein PLM training
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/local/neuware/lib64/libcnrt.so
cd "$(dirname "$0")"
exec /torch/venv3/pytorch/bin/python3 train.py \
    --data data/swissprot.fasta \
    --max_len 512 \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 5 \
    --warmup_steps 1000 \
    --save_every 5000 \
    --eval_every 2000 \
    --out_dir output \
    --seed 42
