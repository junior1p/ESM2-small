"""
Protein Language Model Training on MLU
Architecture: ESM-2 style, ~9.6M parameters
- 12 layers, d_model=256, nhead=8
- Masked Language Modeling (MLM) objective
Features: AMP-free (MLU370 float32), WandB,断点续训, best ckpt, eval
"""

import os
import sys
import math
import time
import random
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ===== MLU Setup =====
os.environ['LD_LIBRARY_PATH'] = '/usr/local/neuware/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_PRELOAD'] = '/usr/local/neuware/lib64/libcnrt.so'

DEVICE = 'mlu'

# ===== Tokenizer =====
class ProteinTokenizer:
    """Amino acid level tokenizer — 31 tokens (20 AA + 5 rare + 6 special)"""
    AA = "ACDEFGHIKLMNPQRSTVWY"
    RARE = "XOUBZ"
    SPECIAL = ["<MSA>", "<gap>", "<mask>", "<unk>", "<pad>", "<eos>"]

    def __init__(self):
        self.tok_to_id = {}
        self.id_to_tok = {}
        for i, aa in enumerate(self.AA):
            self.tok_to_id[aa] = i
            self.id_to_tok[i] = aa
        offset = len(self.AA)
        for i, r in enumerate(self.RARE):
            self.tok_to_id[r] = offset + i
            self.id_to_tok[offset + i] = r
        for i, s in enumerate(self.SPECIAL):
            self.tok_to_id[s] = offset + len(self.RARE) + i
            self.id_to_tok[offset + len(self.RARE) + i] = s
        self.vocab_size = len(self.tok_to_id)  # 31
        self.mask_tok = self.tok_to_id["<mask>"]
        self.unk_tok = self.tok_to_id["<unk>"]
        self.pad_tok = self.tok_to_id["<pad>"]
        self.eos_tok = self.tok_to_id["<eos>"]
        self.aa_set = set(self.AA)

    def encode(self, seq):
        return [self.tok_to_id.get(aa, self.unk_tok) for aa in seq.upper()]

    def decode(self, ids):
        return ''.join(self.id_to_tok.get(i, '<unk>') for i in ids)


# ===== Dataset =====
class ProteinDataset(Dataset):
    def __init__(self, fasta_path, tokenizer, max_len=512, min_len=50):
        print(f"Loading sequences from {fasta_path}...")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_len = min_len
        self.sequences = self._load_fasta(fasta_path)
        print(f"Loaded {len(self.sequences)} sequences (len {min_len}-{max_len})")

    def _load_fasta(self, path):
        seqs = []
        current_seq = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        seq_str = ''.join(current_seq)
                        if self.min_len <= len(seq_str) <= self.max_len:
                            valid = sum(1 for c in seq_str if c in self.tokenizer.aa_set)
                            if valid / len(seq_str) >= 0.8:
                                seqs.append(seq_str)
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                seq_str = ''.join(current_seq)
                if self.min_len <= len(seq_str) <= self.max_len:
                    valid = sum(1 for c in seq_str if c in self.tokenizer.aa_set)
                    if valid / len(seq_str) >= 0.8:
                        seqs.append(seq_str)
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = self.tokenizer.encode(seq)

        if len(tokens) > self.max_len:
            start = random.randint(0, len(tokens) - self.max_len)
            tokens = tokens[start:start + self.max_len]

        labels = torch.tensor(tokens, dtype=torch.long)
        tokens = torch.tensor(tokens, dtype=torch.long)
        mask_prob = torch.rand(len(tokens))
        masked_tokens = tokens.clone()
        is_masked = mask_prob < 0.15
        masked_tokens[is_masked] = self.tokenizer.mask_tok

        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            masked_tokens = torch.cat([masked_tokens, torch.full((pad_len,), self.tokenizer.pad_tok)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
            is_masked = torch.cat([is_masked, torch.zeros(pad_len, dtype=torch.bool)])

        return masked_tokens, labels, is_masked


# ===== Model =====
class ESM2Small(nn.Module):
    """
    ESM-2 style model, ~9.6M params:
    - 12 TransformerEncoder layers, d_model=256, nhead=8
    - FFN dim=1024, Pre-norm, GELU, dropout=0.1
    """
    def __init__(self, vocab_size=31, max_len=512):
        super().__init__()
        d_model = 256
        nhead = 8
        nlayer = 12
        ffn_dim = 1024

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True
            )
            for _ in range(nlayer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens, mask=None):
        B, L = tokens.shape
        x = self.embed(tokens) + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return self.proj(x)


# ===== WandB Setup (lazy) =====
def setup_wandb(config, args):
    try:
        import wandb
        wandb_key = os.environ.get('WANDB_API_KEY', '')
        if wandb_key:
            wandb.login(key=wandb_key)
            run = wandb.init(
                project=args.wandb_project or 'protein-plm',
                name=args.wandb_name or f'ESM2-small-{int(time.time())}',
                config=config,
                resume='allow',
            )
            return run
        else:
            print("WandB: no API key found, skipping")
            return None
    except ImportError:
        print("WandB: not installed, skipping")
        return None


# ===== Training =====
def train_epoch(model, loader, optimizer, scheduler, device, epoch, wandb_run, log_every=100):
    model.train()
    total_loss = 0.0
    total_masked = 0
    t0 = time.time()

    for step, (tokens, labels, is_masked) in enumerate(loader):
        tokens = tokens.to(device)
        labels = labels.to(device)
        mask = (tokens == 0)  # padding mask

        logits = model(tokens)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        n_masked = is_masked.sum().item()
        total_loss += loss.item() * n_masked
        total_masked += n_masked

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            avg_loss = total_loss / total_masked
            ppl = math.exp(min(avg_loss, 10))
            throughput = log_every * tokens.size(0) * tokens.size(1) / elapsed / 1000
            eta = (len(loader) - step - 1) * elapsed / log_every / 3600

            log_str = (f"  Step {step+1:4d} | loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                       f"lr={lr:.2e} | throughput={throughput:.0f}K tok/s | eta={eta:.1f}h")
            print(log_str)

            if wandb_run:
                wandb_run.log({
                    'train/loss': avg_loss,
                    'train/ppl': ppl,
                    'train/lr': lr,
                    'train/throughput': throughput,
                    'epoch': epoch + 1,
                    'step': epoch * len(loader) + step + 1,
                })

            total_loss = 0.0
            total_masked = 0
            t0 = time.time()

    avg_loss = total_loss / max(total_masked, 1)
    return avg_loss, math.exp(min(avg_loss, 10))


def evaluate(model, loader, device, epoch=0, wandb_run=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for tokens, labels, is_masked in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            valid = (labels != -100).sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 10))

    log_str = f"  Val | loss={avg_loss:.4f} | ppl={ppl:.1f}"
    print(log_str)

    if wandb_run:
        wandb_run.log({
            'val/loss': avg_loss,
            'val/ppl': ppl,
            'epoch': epoch + 1,
        })

    return avg_loss, ppl


# ===== Fitness Evaluation (mutation prediction) =====
def evaluate_fitness(model, tokenizer, device, wandb_run=None):
    """
    Evaluate on a small set of log-odds mutation effects.
    Uses masked token prediction to score mutations.
    Runs on CPU to avoid MLU triton inference kernel issues.
    """
    import copy
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    # Human HbS (sickle cell) - common variant benchmarks
    # Format: (wt_seq, pos, mut_aa, expected_effect)
    benchmarks = [
        # Thermostability & fluorescence benchmarks
        ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH", 7, "V", "neutral"),
        ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH", 7, "I", "neutral"),
        ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH", 80, "L", "neutral"),
        # GFP chromophore mutants (representative)
        ("VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQC", 66, "Y", "brighter"),
        ("VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQC", 66, "H", "dimer"),
    ]

    print("\n=== Fitness Evaluation (CPU inference) ===")
    results = []
    for wt_seq, pos, mut_aa, effect in benchmarks:
        if pos >= len(wt_seq):
            continue
        wt_aa = wt_seq[pos]
        if wt_aa == mut_aa:
            continue

        wt_ids = tokenizer.encode(wt_seq)
        wt_ids[pos] = tokenizer.mask_tok
        wt_tensor = torch.tensor([wt_ids], dtype=torch.long)
        with torch.no_grad():
            logits = model_cpu(wt_tensor)
            mask_logit = logits[0, pos, :]
            wt_aa_id = tokenizer.tok_to_id.get(wt_aa, -1)
            mut_aa_id = tokenizer.tok_to_id.get(mut_aa, -1)
            if wt_aa_id < 0 or mut_aa_id < 0:
                continue
            wt_score = mask_logit[wt_aa_id].item()
            mut_score = mask_logit[mut_aa_id].item()
            delta = mut_score - wt_score

        results.append({
            'wt_aa': wt_aa, 'pos': pos, 'mut_aa': mut_aa,
            'effect': effect,
            'wt_score': wt_score, 'mut_score': mut_score,
            'delta': delta
        })
        print(f"  {wt_aa}{pos}{mut_aa} ({effect:10s}) | WT={wt_score:.2f} | Mut={mut_score:.2f} | Δ={delta:.2f}")

    if len(results) >= 3:
        effect_map = {'brighter': 1, 'neutral': 0, 'dimer': -1, 'deleterious': -2}
        pred_scores = [r['delta'] for r in results]
        true_scores = [effect_map.get(r['effect'], 0) for r in results]
        spearman = compute_spearman(pred_scores, true_scores)
        print(f"  Spearman ρ: {spearman:.3f}")
        if wandb_run:
            wandb_run.log({'fitness/spearman': spearman})

    del model_cpu
    return results


def compute_spearman(pred, true):
    n = len(pred)
    if n < 2:
        return 0.0
    # Simple rank correlation
    def rank(x):
        return sorted(range(len(x)), key=lambda i: x[i])
    pred_ranks = [sorted(range(n), key=lambda i: pred[i])[i] for i in range(n)]
    true_ranks = [sorted(range(n), key=lambda i: true[i])[i] for i in range(n)]
    mean_p = sum(pred_ranks) / n
    mean_t = sum(true_ranks) / n
    num = sum((pred_ranks[i] - mean_p) * (true_ranks[i] - mean_t) for i in range(n))
    den = math.sqrt(sum((p - mean_p)**2 for p in pred_ranks) * sum((t - mean_t)**2 for t in true_ranks))
    return num / den if den > 0 else 0.0


# ===== Checkpoint =====
def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    val_loss, config, path, is_best=False):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': config,
    }, path)
    print(f"  ✓ Saved: {path}")
    if is_best:
        best_path = path.replace('.pt', '_best.pt')
        shutil.copy2(path, best_path)
        print(f"  ✓ Best: {best_path}")


def load_checkpoint(path, model, optimizer, scheduler):
    print(f"  Resuming from: {path}")
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt['epoch']
    global_step = ckpt['global_step']
    val_loss = ckpt.get('val_loss', float('inf'))
    print(f"  Resumed: epoch={epoch}, global_step={global_step}, val_loss={val_loss:.4f}")
    return epoch, global_step, val_loss


# ===== Main =====
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/root/protein_plm/data/swissprot.fasta')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--out_dir', type=str, default='/root/protein_plm/output')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint.pt to resume from')
    parser.add_argument('--wandb_project', type=str, default='protein-plm-mlu')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"max_len={args.max_len}, lr={args.lr}")

    # Tokenizer
    tokenizer = ProteinTokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Dataset
    full_ds = ProteinDataset(args.data, tokenizer, max_len=args.max_len, min_len=50)
    n_train = int(len(full_ds) * 0.95)
    n_val = len(full_ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = ESM2Small(vocab_size=vocab_size, max_len=args.max_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    config = {
        'n_params': n_params,
        'vocab_size': vocab_size,
        'max_len': args.max_len,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'n_train': n_train,
        'n_val': n_val,
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Optimizer
    no_decay = ['bias', 'norm']
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    wandb_run = None

    if args.resume and os.path.exists(args.resume):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        # Recompute scheduler's step count
        for _ in range(global_step):
            scheduler.step()

    # WandB
    wandb_run = setup_wandb(config, args)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training from epoch {start_epoch+1}, global_step {global_step}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Evaluate at epoch start
        val_loss, val_ppl = evaluate(model, val_loader, device, epoch, wandb_run)
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Periodic checkpoint
        ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch{epoch+1}.pt')
        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                        val_loss, config, ckpt_path, is_best=is_best)

        # Train epoch
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_masked = 0

        for step, (tokens, labels, is_masked) in enumerate(train_loader):
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = model(tokens)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            n_masked = is_masked.sum().item()
            epoch_loss += loss.item() * n_masked
            epoch_masked += n_masked

            # Log
            if (step + 1) % 100 == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / max(epoch_masked, 1)
                ppl = math.exp(min(avg_loss, 10))
                throughput = 100 * tokens.size(0) * tokens.size(1) / elapsed / 1000
                eta = (len(train_loader) - step - 1) * elapsed / 100 / 3600
                print(f"  Step {step+1:4d} | loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                      f"lr={lr:.2e} | throughput={throughput:.0f}K tok/s | eta={eta:.1f}h")
                if wandb_run:
                    wandb_run.log({
                        'train/loss': avg_loss,
                        'train/ppl': ppl,
                        'train/lr': lr,
                        'train/throughput': throughput,
                        'epoch': epoch + 1,
                        'step': global_step,
                    })
                epoch_loss = 0.0
                epoch_masked = 0
                t0 = time.time()

            # Periodic eval + checkpoint
            if global_step > 0 and global_step % args.eval_every == 0:
                val_loss2, val_ppl2 = evaluate(model, val_loader, device, epoch, wandb_run)
                ckpt_path = os.path.join(args.out_dir, f'checkpoint_step{global_step}.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                val_loss2, config, ckpt_path, is_best=(val_loss2 < best_val_loss))
                if val_loss2 < best_val_loss:
                    best_val_loss = val_loss2
                model.train()

        # End of epoch
        print(f"\n  Train (epoch avg) | loss={epoch_loss/max(epoch_masked,1):.4f}")

        if wandb_run:
            wandb_run.log({'epoch': epoch + 1})

    # Final fitness eval
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("="*60)

    evaluate_fitness(model, tokenizer, device, wandb_run)

    # Final checkpoint
    final_path = os.path.join(args.out_dir, 'checkpoint_final.pt')
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, global_step,
                    best_val_loss, config, final_path, is_best=True)

    if wandb_run:
        wandb_run.finish()

    print("\nDone!")


if __name__ == '__main__':
    main()
