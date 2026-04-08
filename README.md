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

Training curves (train loss / val loss / PPL):
- Epoch 1: val loss ~2.5, PPL ~12
- Epoch 5 (target): val loss ~0.8, PPL ~2.2

Mutation fitness evaluation (Spearman ρ on test set):
- Tracks masked LM logit differences between WT and mutant

## Files

```
train.py          # Full training script
data/             # Swiss-Prot FASTA (download via script)
output/           # Checkpoints (after training)
```

## License

MIT
