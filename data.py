"""
data.py
=======
ProteinDataset and collate_fn for ESM-2 style masked language modeling.
"""

from __future__ import annotations

import functools
import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import ESMTokenizer

# Standard amino acid single-letter codes (used for filtering)
_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Token IDs for standard amino acids in ESMTokenizer (indices 4-28 inclusive)
_AA_TOKEN_RANGE = (4, 29)  # [4, 28] inclusive -> randint(4, 28)


class ProteinDataset(Dataset):
    """Dataset for ESM-2 style masked language modeling on protein sequences.

    Parameters
    ----------
    fasta_path : str
        Path to a FASTA file containing protein sequences.
    tokenizer : ESMTokenizer
        Tokenizer instance (33-token ESM-2 vocabulary).
    max_len : int, optional
        Maximum token length including <cls> and <eos> (default: 512).
    min_len : int, optional
        Minimum sequence length in amino acids, excluding special tokens
        (default: 10).
    mask_prob : float, optional
        Probability of masking each non-special token for MLM (default: 0.15).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        fasta_path: str,
        tokenizer: ESMTokenizer,
        max_len: int = 512,
        min_len: int = 10,
        mask_prob: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_len = min_len
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)

        self.sequences: List[str] = self._load_fasta(fasta_path)
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences found in {fasta_path}")

    # ------------------------------------------------------------------
    # FASTA loading
    # ------------------------------------------------------------------

    def _load_fasta(self, path: str) -> List[str]:
        """Parse a FASTA file and return filtered protein sequences.

        Filters applied:
        - Length: min_len <= len(seq) <= max_len - 2
        - Non-standard amino acid ratio: sequences with >20% non-standard
          characters are discarded.
        """
        sequences: List[str] = []
        max_aa_len = self.max_len - 2  # reserve 2 positions for <cls> and <eos>

        current_seq_parts: List[str] = []

        def _flush(parts: List[str]) -> None:
            if not parts:
                return
            seq = "".join(parts).upper().strip()
            if len(seq) < self.min_len or len(seq) > max_aa_len:
                return
            non_std = sum(1 for aa in seq if aa not in _STANDARD_AA)
            if non_std / len(seq) > 0.20:
                return
            sequences.append(seq)

        with open(path, "r") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    _flush(current_seq_parts)
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)
            _flush(current_seq_parts)

        return sequences

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single masked training example.

        Steps:
        1. Retrieve sequence; randomly crop if longer than max_len - 2.
        2. Encode with tokenizer (adds <cls> and <eos>).
        3. Apply MLM masking to non-special token positions:
           - 80% -> <mask> token
           - 10% -> random amino acid token (id in [4, 28])
           - 10% -> unchanged
        4. Build labels: masked positions keep original id; others are -100.
        5. Return (input_ids, labels, attention_mask) as LongTensors.
        """
        seq = self.sequences[idx]
        max_aa_len = self.max_len - 2

        # Random crop if necessary
        if len(seq) > max_aa_len:
            start = self.rng.randint(0, len(seq) - max_aa_len)
            seq = seq[start: start + max_aa_len]

        # Encode: [cls_id, aa_id_1, ..., aa_id_n, eos_id]
        token_ids: List[int] = self.tokenizer.encode(seq)
        L = len(token_ids)

        special_ids = self.tokenizer.get_special_token_ids()

        input_ids = list(token_ids)
        labels = [-100] * L

        for i, tid in enumerate(token_ids):
            if tid in special_ids:
                continue
            if self.rng.random() < self.mask_prob:
                labels[i] = tid
                r = self.rng.random()
                if r < 0.80:
                    # 80%: replace with <mask>
                    input_ids[i] = self.tokenizer.mask_token_id
                elif r < 0.90:
                    # 10%: replace with random amino acid token (4-28 inclusive)
                    input_ids[i] = self.rng.randint(
                        _AA_TOKEN_RANGE[0], _AA_TOKEN_RANGE[1] - 1
                    )
                # else: 10% keep original

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        attention_mask_t = torch.ones(L, dtype=torch.long)

        return input_ids_t, labels_t, attention_mask_t


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_token_id: int = 1,
) -> dict:
    """Pad a batch of (input_ids, labels, attention_mask) to the same length.

    Parameters
    ----------
    batch : list of (input_ids, labels, attention_mask)
        Each element is a tuple of 1-D LongTensors.
    pad_token_id : int, optional
        Token ID used to pad input_ids (default: 1 = <pad>).

    Returns
    -------
    dict
        {"input_ids": Tensor, "labels": Tensor, "attention_mask": Tensor}
        each of shape (B, max_len).
    """
    input_ids_list, labels_list, attn_mask_list = zip(*batch)

    max_len = max(t.size(0) for t in input_ids_list)

    padded_input_ids = []
    padded_labels = []
    padded_attn_masks = []

    for ids, lbl, mask in zip(input_ids_list, labels_list, attn_mask_list):
        pad_len = max_len - ids.size(0)
        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        )
        padded_attn_masks.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_attn_masks),
    }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------


def build_dataloaders(
    train_fasta: str,
    val_fasta: str,
    tokenizer: ESMTokenizer,
    batch_size: int = 32,
    max_len: int = 512,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for MLM pre-training.

    Parameters
    ----------
    train_fasta : str
        Path to the training FASTA file.
    val_fasta : str
        Path to the validation FASTA file.
    tokenizer : ESMTokenizer
        Tokenizer instance.
    batch_size : int, optional
        Batch size (default: 32).
    max_len : int, optional
        Maximum token length including <cls>/<eos> (default: 512).
    num_workers : int, optional
        Number of DataLoader worker processes (default: 4).
    seed : int, optional
        Random seed (default: 42).

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    train_dataset = ProteinDataset(
        fasta_path=train_fasta,
        tokenizer=tokenizer,
        max_len=max_len,
        seed=seed,
    )
    val_dataset = ProteinDataset(
        fasta_path=val_fasta,
        tokenizer=tokenizer,
        max_len=max_len,
        seed=seed,
    )

    _collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
