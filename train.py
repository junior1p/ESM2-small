"""
train.py
========
ESM-2 style protein language model training.
Supports CUDA (BF16 AMP), Cambricon MLU370 (FP32), and CPU.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler

# ---------------------------------------------------------------------------
# MLU support (optional)
# ---------------------------------------------------------------------------
try:
    import torch_mlu  # noqa: F401
    _MLU_AVAILABLE = hasattr(torch, "mlu") and torch.mlu.is_available()
except ImportError:
    _MLU_AVAILABLE = False

# ---------------------------------------------------------------------------
# wandb support (optional)
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import ESMTokenizer
from model import build_model
from data import ProteinDataset, collate_fn, build_dataloaders


# ===========================================================================
# Device detection
# ===========================================================================

def auto_detect_device(requested: str = "auto") -> torch.device:
    """Detect the best available compute device.

    Priority: cuda > mlu > cpu

    Parameters
    ----------
    requested : str, optional
        One of "auto", "cuda", "mlu", "cpu" (default: "auto").

    Returns
    -------
    torch.device
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _MLU_AVAILABLE:
            return torch.device("mlu")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "mlu":
        if not _MLU_AVAILABLE:
            raise RuntimeError("MLU requested but not available.")
        return torch.device("mlu")
    return torch.device("cpu")


# ===========================================================================
# AMP context
# ===========================================================================

def get_amp_context(device: torch.device):
    """Return AMP configuration for the given device.

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    Tuple[bool, Optional[torch.dtype], Optional[GradScaler]]
        (use_amp, amp_dtype, scaler)
        - CUDA: use_amp=True, dtype=torch.bfloat16, scaler=GradScaler()
        - MLU/CPU: use_amp=False, dtype=None, scaler=None
    """
    if device.type == "cuda":
        return True, torch.bfloat16, GradScaler()
    return False, None, None


# ===========================================================================
# EMA
# ===========================================================================

class EMA:
    """Exponential Moving Average of model weights.

    Parameters
    ----------
    model : nn.Module
        The model whose weights to track.
    decay : float, optional
        EMA decay factor (default: 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict = {}
        self._backup: dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.float()
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Copy EMA weights into model (for eval/save)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.data.dtype))

    def restore(self, model: nn.Module) -> None:
        """Restore original weights after apply_shadow."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state["shadow"]
        self.decay = state["decay"]


# ===========================================================================
# Scheduler
# ===========================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# Training epoch
# ===========================================================================

def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    global_step: int,
    wandb_run,
    use_amp: bool,
    amp_dtype,
    scaler: Optional[GradScaler],
    grad_accum: int = 1,
    ema: Optional[EMA] = None,
    log_every: int = 100,
) -> Tuple[float, int]:
    """Run one training epoch.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : Optimizer
    scheduler : LRScheduler
    device : torch.device
    epoch : int
    global_step : int
    wandb_run : wandb run or None
    use_amp : bool
    amp_dtype : torch.dtype or None
    scaler : GradScaler or None
    grad_accum : int
    ema : EMA or None
    log_every : int

    Returns
    -------
    Tuple[float, int]
        (avg_loss, global_step)
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()

    num_batches = len(loader)
    epoch_start = time.time()
    step_start = time.time()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Count valid (non-ignored) label tokens for throughput
        n_tokens = (labels != -100).sum().item()

        # Forward pass with optional AMP
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_dtype is not None else torch.float32,
            enabled=use_amp,
        ):
            logits = model(input_ids)  # (B, L, vocab_size)
            # Flatten for cross-entropy
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            # Scale loss for gradient accumulation
            loss_scaled = loss / grad_accum

        # Backward
        if use_amp and scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        total_loss += loss.item()
        total_tokens += n_tokens

        # Optimizer step every grad_accum micro-steps
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == num_batches:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # EMA update
            if ema is not None:
                ema.update(model)

            # Logging
            if global_step % log_every == 0:
                elapsed = time.time() - step_start
                avg_loss_so_far = total_loss / (batch_idx + 1)
                ppl = math.exp(min(avg_loss_so_far, 20))
                lr = scheduler.get_last_lr()[0]
                throughput = total_tokens / max(elapsed, 1e-6)
                batches_remaining = num_batches - batch_idx - 1
                eta_sec = batches_remaining * (elapsed / max(batch_idx + 1, 1))

                print(
                    f"Epoch {epoch:3d} | Step {global_step:7d} | "
                    f"loss {avg_loss_so_far:.4f} | ppl {ppl:.2f} | "
                    f"lr {lr:.2e} | "
                    f"thr {throughput:.0f} tok/s | "
                    f"eta {eta_sec/60:.1f}m"
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss_so_far,
                            "train/ppl": ppl,
                            "train/lr": lr,
                            "train/throughput": throughput,
                            "global_step": global_step,
                        }
                    )

                step_start = time.time()
                total_tokens = 0

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    amp_dtype,
    epoch: int = 0,
    wandb_run=None,
) -> Tuple[float, float]:
    """Evaluate the model on a validation DataLoader.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    use_amp : bool
    amp_dtype : torch.dtype or None
    epoch : int
    wandb_run : wandb run or None

    Returns
    -------
    Tuple[float, float]
        (avg_loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=use_amp,
            ):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))

    print(f"  [Eval] epoch={epoch} | loss={avg_loss:.4f} | ppl={perplexity:.2f}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "val/loss": avg_loss,
                "val/ppl": perplexity,
                "epoch": epoch,
            }
        )

    return avg_loss, perplexity


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer,
    scheduler,
    ema: Optional[EMA],
    val_loss: float,
    config: dict,
    model_size: str,
    device: torch.device,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path : str
        File path to save the checkpoint.
    epoch : int
    global_step : int
    model : nn.Module
    optimizer : Optimizer
    scheduler : LRScheduler
    ema : EMA or None
    val_loss : float
    config : dict
    model_size : str
    device : torch.device
    """
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
    }

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "val_loss": val_loss,
        "config": config,
        "model_size": model_size,
        "rng_state": rng_state,
    }

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(checkpoint, path)
    print(f"  [Checkpoint] Saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    ema: Optional[EMA] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a training checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    model : nn.Module
    optimizer : Optimizer or None
    scheduler : LRScheduler or None
    ema : EMA or None
    device : torch.device

    Returns
    -------
    dict
        The full checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    # Restore RNG state
    rng = checkpoint.get("rng_state", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"])

    print(
        f"  [Checkpoint] Loaded from {path} "
        f"(epoch={checkpoint.get('epoch', '?')}, "
        f"step={checkpoint.get('global_step', '?')})"
    )
    return checkpoint


# ===========================================================================
# Built-in GFP fitness evaluation
# ===========================================================================

# avGFP WT sequence (first 100 aa for fast evaluation)
_GFP_WT_SEQ = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQ"
    "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDG"
)

_GFP_MUTATIONS = [
    {"pos": 64, "wt": "S", "mut": "T", "effect": "brighter"},   # S65T (0-indexed: 64)
    {"pos": 63, "wt": "F", "mut": "L", "effect": "neutral"},    # F64L (0-indexed: 63)
    {"pos": 65, "wt": "Y", "mut": "W", "effect": "brighter"},   # Y66W (0-indexed: 65)
    {"pos": 65, "wt": "Y", "mut": "C", "effect": "deleterious"},# Y66C (0-indexed: 65)
]


def evaluate_fitness_builtin(
    model: nn.Module,
    tokenizer: ESMTokenizer,
    device: torch.device,
) -> None:
    """Evaluate model fitness using masked logit-diff on avGFP mutations.

    Uses per-position masked logit-diff: mask the mutation site, compute
    log P(mut) - log P(wt) from the model's output logits.

    Parameters
    ----------
    model : nn.Module
    tokenizer : ESMTokenizer
    device : torch.device
    """
    print("\n[GFP Fitness Evaluation]")
    print(f"  WT sequence (first 100aa): {_GFP_WT_SEQ[:30]}...")

    model.eval()
    wt_seq = _GFP_WT_SEQ

    results = []
    with torch.no_grad():
        for mut_info in _GFP_MUTATIONS:
            pos = mut_info["pos"]       # 0-indexed in sequence
            wt_aa = mut_info["wt"]
            mut_aa = mut_info["mut"]
            effect = mut_info["effect"]

            # Verify WT residue
            if wt_seq[pos] != wt_aa:
                print(
                    f"  WARNING: Expected {wt_aa} at pos {pos}, "
                    f"got {wt_seq[pos]}. Skipping."
                )
                continue

            # Build masked sequence: replace pos with <mask>
            masked_seq = wt_seq[:pos] + "<mask>" + wt_seq[pos + 1:]

            # Tokenize: encode WT but replace the residue position with mask token
            # token position = pos + 1 (offset by <cls>)
            token_ids = tokenizer.encode(wt_seq)
            token_pos = pos + 1  # +1 for <cls>

            masked_ids = list(token_ids)
            masked_ids[token_pos] = tokenizer.mask_token_id

            input_tensor = torch.tensor([masked_ids], dtype=torch.long).to(device)

            logits = model(input_tensor)  # (1, L, vocab_size)
            log_probs = torch.log_softmax(logits[0, token_pos], dim=-1)

            wt_id = tokenizer.encode(wt_aa)[1]   # skip <cls>
            mut_id = tokenizer.encode(mut_aa)[1]  # skip <cls>

            score = (log_probs[mut_id] - log_probs[wt_id]).item()
            results.append(
                {
                    "mutation": f"{wt_aa}{pos + 1}{mut_aa}",
                    "score": score,
                    "effect": effect,
                }
            )
            print(
                f"  {wt_aa}{pos + 1}{mut_aa:3s} | score={score:+.4f} | "
                f"expected={effect}"
            )

    # Summary
    if results:
        brighter = [r["score"] for r in results if r["effect"] == "brighter"]
        deleterious = [r["score"] for r in results if r["effect"] == "deleterious"]
        neutral = [r["score"] for r in results if r["effect"] == "neutral"]

        print("\n  Summary:")
        if brighter:
            print(f"    Brighter mutations avg score:    {sum(brighter)/len(brighter):+.4f}")
        if neutral:
            print(f"    Neutral mutations avg score:     {sum(neutral)/len(neutral):+.4f}")
        if deleterious:
            print(f"    Deleterious mutations avg score: {sum(deleterious)/len(deleterious):+.4f}")

        # Sanity check: brighter > deleterious?
        if brighter and deleterious:
            ok = sum(brighter) / len(brighter) > sum(deleterious) / len(deleterious)
            print(f"    Rank order correct (brighter > deleterious): {ok}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESM-2 style protein language model training."
    )

    # Data
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training FASTA file.")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation FASTA file.")

    # Model
    parser.add_argument("--model_size", type=str, default="8M",
                        choices=["8M", "35M", "150M"],
                        help="Model size configuration (default: 8M).")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum sequence length including cls/eos (default: 512).")

    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device batch size (default: 32).")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate (default: 1e-4).")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5).")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="LR warmup steps (default: 1000).")

    # Checkpointing / evaluation
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N steps (default: 5000).")
    parser.add_argument("--eval_every", type=int, default=2000,
                        help="Evaluate every N steps mid-epoch (default: 2000).")
    parser.add_argument("--out_dir", type=str, default="output",
                        help="Output directory for checkpoints (default: output).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (default: None).")

    # Device / precision
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mlu, cpu (default: auto).")

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="EMA decay (0 = disabled, default: 0.0).")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="protein-plm",
                        help="W&B project name (default: protein-plm).")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="W&B run name (default: None).")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Device & AMP
    # ------------------------------------------------------------------
    device = auto_detect_device(args.device)
    use_amp, amp_dtype, scaler = get_amp_context(device)
    print(f"Device: {device} | AMP: {use_amp} | dtype: {amp_dtype}")

    # ------------------------------------------------------------------
    # Tokenizer & Model
    # ------------------------------------------------------------------
    tokenizer = ESMTokenizer()
    model = build_model(model_size=args.model_size, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model: ESM-2 {args.model_size} | Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        train_fasta=args.train_data,
        val_fasta=args.val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=4,
        seed=args.seed,
    )
    print(
        f"Data: {len(train_loader.dataset)} train seqs, "
        f"{len(val_loader.dataset)} val seqs"
    )

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    total_steps = (
        len(train_loader) // args.grad_accum * args.epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    ema: Optional[EMA] = None
    if args.ema_decay > 0.0:
        ema = EMA(model, decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wandb_run = None
    if _WANDB_AVAILABLE:
        try:
            wandb_run = _wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        ckpt = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema, device
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # ------------------------------------------------------------------
    # Config dict for checkpoints
    # ------------------------------------------------------------------
    config = vars(args)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Evaluate at the start of each epoch
        val_loss, val_ppl = evaluate(
            model, val_loader, device, use_amp, amp_dtype,
            epoch=epoch, wandb_run=wandb_run
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=os.path.join(args.out_dir, "best_checkpoint.pt"),
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                val_loss=val_loss,
                config=config,
                model_size=args.model_size,
                device=device,
            )

        # Train one epoch (with mid-epoch eval and step checkpoints)
        def _mid_epoch_hooks(step: int, model: nn.Module) -> None:
            nonlocal best_val_loss
            if step % args.eval_every == 0 and step > 0:
                vl, vp = evaluate(
                    model, val_loader, device, use_amp, amp_dtype,
                    epoch=epoch, wandb_run=wandb_run
                )
                if vl < best_val_loss:
                    best_val_loss = vl
                    save_checkpoint(
                        path=os.path.join(args.out_dir, "best_checkpoint.pt"),
                        epoch=epoch,
                        global_step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        ema=ema,
                        val_loss=vl,
                        config=config,
                        model_size=args.model_size,
                        device=device,
                    )
            if step % args.save_every == 0 and step > 0:
                save_checkpoint(
                    path=os.path.join(args.out_dir, f"step_{step:08d}.pt"),
                    epoch=epoch,
                    global_step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    val_loss=best_val_loss,
                    config=config,
                    model_size=args.model_size,
                    device=device,
                )

        # Custom train loop with mid-epoch hooks
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        num_batches = len(train_loader)
        epoch_start = time.time()
        step_start = time.time()
        total_tokens = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            n_tokens = (labels != -100).sum().item()

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=use_amp,
            ):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss_scaled = loss / args.grad_accum

            if use_amp and scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            total_loss += loss.item()
            total_tokens += n_tokens

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == num_batches:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if ema is not None:
                    ema.update(model)

                # Mid-epoch hooks
                _mid_epoch_hooks(global_step, model)

                if global_step % 100 == 0:
                    elapsed = time.time() - step_start
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    ppl = math.exp(min(avg_loss_so_far, 20))
                    lr = scheduler.get_last_lr()[0]
                    throughput = total_tokens / max(elapsed, 1e-6)
                    batches_remaining = num_batches - batch_idx - 1
                    eta_sec = batches_remaining * (elapsed / max(batch_idx + 1, 1))

                    print(
                        f"Epoch {epoch + 1:3d} | Step {global_step:7d} | "
                        f"loss {avg_loss_so_far:.4f} | ppl {ppl:.2f} | "
                        f"lr {lr:.2e} | "
                        f"thr {throughput:.0f} tok/s | "
                        f"eta {eta_sec/60:.1f}m"
                    )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": avg_loss_so_far,
                                "train/ppl": ppl,
                                "train/lr": lr,
                                "train/throughput": throughput,
                                "global_step": global_step,
                            }
                        )

                    step_start = time.time()
                    total_tokens = 0

        avg_train_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1} done | avg_train_loss={avg_train_loss:.4f}")

        # Save epoch checkpoint
        save_checkpoint(
            path=os.path.join(args.out_dir, f"epoch_{epoch + 1:03d}.pt"),
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            val_loss=best_val_loss,
            config=config,
            model_size=args.model_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\n[Final Evaluation]")
    val_loss, val_ppl = evaluate(
        model, val_loader, device, use_amp, amp_dtype,
        epoch=args.epochs, wandb_run=wandb_run
    )
    print(f"Final val_loss={val_loss:.4f} | ppl={val_ppl:.2f}")

    # ------------------------------------------------------------------
    # Built-in GFP fitness evaluation
    # ------------------------------------------------------------------
    evaluate_fitness_builtin(model, tokenizer, device)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
