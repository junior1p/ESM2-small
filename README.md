# ESM2-small

**9.6M parameter protein language model** trained on Swiss-Prot with MLU370 (Cambricon).

Architecture mirrors [ESM-2](https://facebookresearch.github.io/esm/):
- 12-layer Transformer Encoder
- d_model=256, nhead=8, FFN dim=1024
- Pre-norm, GELU activation, 15% MLM masking

## Training

| Parameter | Value |
|---|---|
| Data | Swiss-Prot (456,404 train / 22,821 val) |
| Device | MLU370 (Cambricon) — 1 card |
| Batch | 32 × 512 tokens |
| Speed | ~30K tokens/s |
| Epochs | 5 (~2h/epoch, total ~10h) |
| Optimizer | AdamW (lr=1e-4, warmup=1000 steps, cosine decay) |
| Final val loss | **0.4170** |

### Training Progress

| Epoch | Val Loss | Notes |
|---|---|---|
| 1 | 0.4195 | checkpoint ~38MB (EMA) |
| 2 | 0.4235 | — |
| 3 | 0.4182 | best so far |
| 4 | 0.4185 | — |
| 5 | 0.4179 | final epoch |
| **Final** | **0.4170** | checkpoint_final_best.pt |

## Quick Start

```python
import torch
import train

# Load tokenizer & model
tokenizer = train.get_tokenizer()
model = train.ESM2Small(vocab_size=31, max_len=512)
ckpt = torch.load("weights/model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Encode a protein sequence
seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
ids = tokenizer.encode(seq)
tokens = torch.tensor([ids])

# Masked prediction
with torch.no_grad():
    logits = model(tokens)
    print(tokenizer.decode(logits[0].argmax(dim=-1).tolist()))
```

## Zero-Shot Mutation Fitness

Evaluated on GFP fluorescence mutations (4 test cases, CPU inference):

| Mutation | Type | WT score | Mut score | Δ |
|---|---|---|---|---|
| K7V | neutral | 1.39 | 1.52 | +0.13 |
| K7I | neutral | 1.39 | 0.40 | -0.99 |
| G66Y | brighter | 1.70 | 0.33 | -1.37 |
| G66H | dimer | 1.70 | 0.32 | -1.38 |

Spearman ρ = **0.200** (zero-shot, no fine-tuning)

## Files

```
weights/
  model.pt          # Final best checkpoint (110MB)
  config.json       # Training config
train.py            # Full training script
download_data.py    # Swiss-Prot data downloader
requirements.txt    # Python dependencies
start_training.sh   # Training launcher
training.log        # Full training log
fitness_results.txt # Mutation prediction results
data/               # Swiss-Prot FASTA (download via script)
```

## Training from Scratch

```bash
# 1. Download training data
python download_data.py

# 2. Train (MLU370)
bash start_training.sh

# 3. Resume if interrupted
python train.py --resume output/checkpoint_epoch2.pt
```

## License

MIT
