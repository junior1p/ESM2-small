# ESM2-small

**9.6M parameter protein language model** trained on Swiss-Prot with MLU370 (Cambricon).

Architecture mirrors [ESM-2](https://facebookresearch.github.io/esm/):
- 12-layer Transformer Encoder
- d_model=256, nhead=8, FFN dim=1024
- Pre-norm, GELU activation, 15% MLM masking

## Training

- **Data**: [Swiss-Prot](https://rest.uniprot.org) curated protein sequences (456,404 train / 22,821 val)
- **Device**: MLU370 (Cambricon) — 1 card
- **Batch**: 32 × 512 tokens = 16K tokens/step
- **Speed**: ~2 steps/s (~30K tokens/s)
- **Epochs**: 5 epochs (~1.8h/epoch)
- **Optimizer**: AdamW (lr=1e-4, warmup=1000 steps, cosine decay)
- **Output**: `checkpoint_final.pt` + per-epoch checkpoints

## Quick Start

```python
import torch, train

# Load tokenizer & model
tokenizer = train.ProteinTokenizer()
model = train.ESM2Small(vocab_size=31, max_len=512)
ckpt = torch.load("checkpoint_final.pt", map_location="cpu")
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

## Results

Training progress (Step 1100, epoch 1 of 5):
- Loss: 0.45 | PPL: 1.6 | LR: 1e-4 | Speed: ~28K tokens/s
- Epoch 1/5 checkpoint saved after training completes (~1.8h/epoch)

Mutation fitness evaluation (Spearman ρ on test set):
- Tracks masked LM logit differences between WT and mutant

## Files

```
train.py          # Full training script
download_data.py  # Swiss-Prot data downloader
requirements.txt  # Python dependencies
data/             # Swiss-Prot FASTA (download via script)
output/           # Checkpoints (after training):
                    checkpoint_epoch*.pt     # per-epoch ckpts
                    checkpoint_step*.pt      # intermediate ckpts
                    checkpoint_final.pt      # final model
                    checkpoint_final_best.pt # best val loss model
                    config.json              # training config
```

## Training from scratch

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
