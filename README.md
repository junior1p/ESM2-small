# ESM2-small

ESM-2 style protein language model series trained from scratch on Swiss-Prot.

## Model Series

| Model | Layers | d_model | Heads | FFN | Params | Status |
|-------|--------|---------|-------|-----|--------|--------|
| ESM2-8M | 6 | 320 | 20 | 1280 | 7.4M | ✓ Trained |
| ESM2-35M | 6 | 480 | 20 | 1920 | 16.6M | Training ready |
| ESM2-150M | 30 | 640 | 20 | 2560 | 147.6M | Training ready |

## Architecture

- **Rotary Position Embeddings (RoPE)** — replaces learned PE; supports length extrapolation beyond training length
- **Pre-norm Transformer blocks** — LayerNorm applied before attention and FFN sub-layers
- **Weight-tied LM head** — `lm_head.weight = embed_tokens.weight`, reducing parameter count
- **33-token vocabulary** — aligned with ESM-2 official vocab order (includes `<cls>`/`<eos>` boundary tokens)
- **MLM objective** — 15% masking (80% `<mask>` / 10% random token / 10% unchanged)

## Quick Start

### 1. Install

```bash
pip install torch>=2.0 numpy scipy scikit-learn tqdm requests
```

### 2. Download Data

```bash
python scripts/download_data.py --output_dir ./data
# Output: data/swissprot_train.fasta (~433K seqs), data/swissprot_val.fasta (~22K seqs)
```

### 3. Train

```bash
# CUDA (BF16 AMP, recommended)
bash start_training_cuda.sh 8M

# Cambricon MLU370
bash start_training.sh 8M

# Manual (any device, full control)
python train.py \
    --train_data data/swissprot_train.fasta \
    --val_data   data/swissprot_val.fasta \
    --model_size 8M \
    --device     auto \
    --out_dir    output/8M
```

### 4. Evaluate Zero-Shot Fitness

```bash
# Built-in GFP benchmark (5 mutations, no extra data needed)
python scripts/evaluate_fitness.py \
    --checkpoint output/8M/checkpoint_final_best.pt

# ProteinGym CSV (mutant + DMS_score columns)
python scripts/evaluate_fitness.py \
    --checkpoint output/8M/checkpoint_final_best.pt \
    --dms_csv    data/GFP_AVGFP_Sarkisyan2016.csv \
    --output     results/gfp_fitness.csv
```

### 5. Evaluate Embedding Quality

```bash
python scripts/evaluate_embedding.py \
    --checkpoint output/8M/checkpoint_final_best.pt \
    --fasta      data/swissprot_val.fasta \
    --n_seqs     1000 \
    --tsne
```

## Training Details

| Parameter | 8M | 35M | 150M |
|-----------|-----|-----|------|
| Batch size | 32 | 16 | 8 |
| Grad accum | 1 | 2 | 4 |
| Effective batch | 32 | 32 | 32 |
| Learning rate | 1e-4 | 5e-5 | 3e-5 |
| Warmup steps | 1000 | 2000 | 4000 |
| LR schedule | cosine | cosine | cosine |
| EMA | — | 0.999 | 0.999 |
| AMP (CUDA) | BF16 | BF16 | BF16 |

## Hardware

- **CUDA**: Any NVIDIA GPU with BF16 support (Ampere+). A100 recommended for 150M.
- **MLU**: Cambricon MLU370 (FP32, no AMP). Use `start_training.sh`.
- **CPU**: Supported for small models and evaluation only.

## Resume Training

```bash
python train.py \
    --train_data data/swissprot_train.fasta \
    --val_data   data/swissprot_val.fasta \
    --model_size 8M \
    --resume     output/8M/checkpoint_epoch2.pt
```

## Upload Model

```bash
python scripts/upload_to_github.py \
    --repo       junior1p/ESM2-small \
    --checkpoint output/8M/checkpoint_final_best.pt \
    --tag        v2.0-8M \
    --title      "ESM2-8M v2.0 (RoPE, 33-token)" \
    --token      $GITHUB_TOKEN
```

## File Structure

```
ESM2-small/
├── train.py                    # Main training script
├── model.py                    # ESMModel with RoPE (8M/35M/150M)
├── tokenizer.py                # 33-token ESM-2 vocabulary
├── data.py                     # ProteinDataset + MLM collation
├── scripts/
│   ├── download_data.py        # Swiss-Prot downloader
│   ├── evaluate_fitness.py     # Zero-shot mutation fitness
│   ├── evaluate_embedding.py   # Embedding quality (k-NN)
│   └── upload_to_github.py     # Model release uploader
├── configs/
│   ├── 8M.json                 # 8M hyperparameters
│   ├── 35M.json                # 35M hyperparameters
│   └── 150M.json               # 150M hyperparameters
├── start_training.sh           # MLU370 launcher
├── start_training_cuda.sh      # CUDA launcher
└── weights/                    # Legacy 9.6M checkpoint (v1, 31-token)
```

## Legacy Model (v1)

The original `weights/model.pt` is a **9.6M parameter model** trained with a 31-token vocabulary
(no `<cls>` token) on MLU370. It is **not compatible** with the current 33-token tokenizer.
Kept for reference.

- Val loss: 0.4170 | Spearman ρ (GFP, 4 mutations): 0.200

## License

MIT
