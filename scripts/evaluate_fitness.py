"""
evaluate_fitness.py
===================
Zero-shot mutation fitness prediction using a trained ESM-2 style model.
Supports both built-in GFP benchmark and ProteinGym-format CSV files.

Scoring method: per-position masked logit-difference
    score(mut) = log P(mut_aa | context) - log P(wt_aa | context)
    where context has the mutation site masked.

For multi-site mutations (e.g. "A2T:G10S"), scores are summed across sites.

Usage
-----
# Built-in GFP benchmark (no CSV needed):
python evaluate_fitness.py --checkpoint /path/to/checkpoint.pt

# ProteinGym CSV:
python evaluate_fitness.py --checkpoint /path/to/checkpoint.pt \
    --dms_csv /path/to/dms.csv --mutant_col mutant --score_col DMS_score

# Save results:
python evaluate_fitness.py --checkpoint /path/to/checkpoint.pt --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow importing from /workspace/esm2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import build_model
from tokenizer import ESMTokenizer

# ---------------------------------------------------------------------------
# avGFP wild-type sequence (standard 239 aa)
# ---------------------------------------------------------------------------

GFP_WT_SEQ = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQ"
    "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGH"
    "KLYNFERDTNNVFKEIDHAVIDRDHLVNGNKVLENDRQLLNELTKDEVDRMTDELDGDVNGHKFSVSGEG"
    "EGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDD"
)

# ---------------------------------------------------------------------------
# Built-in GFP benchmark mutations
# ---------------------------------------------------------------------------

GFP_MUTATIONS = [
    # (mutation_str, expected_effect)
    # 1-indexed positions, format "WTaaPositionMutaa"
    ("S65T", "brighter"),     # classic brightness-enhancing mutation
    ("F64L", "neutral"),      # neutral mutation
    ("Y66W", "brighter"),     # brightness-enhancing
    ("Y66C", "deleterious"),  # deleterious
    ("K26R", "neutral"),      # neutral
]

# Pseudo-scores for expected effects (used to compute Spearman rho in built-in mode)
_EFFECT_SCORE = {"brighter": 1.0, "neutral": 0.0, "deleterious": -1.0}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def parse_mutation(mut_str: str) -> Tuple[int, str, str]:
    """Parse a 1-indexed mutation string like 'A2T' -> (1, 'A', 'T') (0-indexed pos).

    Parameters
    ----------
    mut_str : str
        Mutation string in format ``WTaaPositionMutaa``, e.g. ``"S65T"``.

    Returns
    -------
    Tuple[int, str, str]
        ``(pos_0indexed, wt_aa, mut_aa)``
    """
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos_1indexed = int(mut_str[1:-1])
    return (pos_1indexed - 1, wt_aa, mut_aa)


def score_mutation(
    model: torch.nn.Module,
    tokenizer: ESMTokenizer,
    device: torch.device,
    wt_seq: str,
    mutations: List[Tuple[int, str, str]],
) -> float:
    """Compute per-position masked logit-difference score for one or more mutations.

    For each mutation site:
      1. Build a masked version of the WT sequence (mask that position).
      2. Run the model to get logits.
      3. Accumulate ``log_softmax[mut_aa] - log_softmax[wt_aa]`` at that position.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ESM-2 style model.
    tokenizer : ESMTokenizer
        Tokenizer aligned with the model vocabulary.
    device : torch.device
        Device to run inference on.
    wt_seq : str
        Wild-type amino-acid sequence (upper-case single-letter codes).
    mutations : List[Tuple[int, str, str]]
        List of ``(pos_0indexed, wt_aa, mut_aa)`` tuples.

    Returns
    -------
    float
        Sum of log-likelihood differences across all mutation sites.
    """
    model.eval()
    total_score = 0.0

    with torch.no_grad():
        for pos_0, wt_aa, mut_aa in mutations:
            # Tokenize WT sequence: [cls, aa1, ..., aaN, eos]
            tokens = tokenizer.encode(wt_seq)
            # Position in token list: pos_0 + 1 (offset by <cls>)
            token_pos = pos_0 + 1
            tokens[token_pos] = tokenizer.mask_token_id

            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(input_ids)  # (1, seq_len, vocab_size)

            # Log-softmax over vocabulary at the masked position
            log_probs = F.log_softmax(logits[0, token_pos, :], dim=-1)

            wt_id = tokenizer._token_to_id.get(wt_aa, tokenizer.unk_token_id)
            mut_id = tokenizer._token_to_id.get(mut_aa, tokenizer.unk_token_id)

            score = (log_probs[mut_id] - log_probs[wt_id]).item()
            total_score += score

    return total_score


def evaluate_from_csv(
    model: torch.nn.Module,
    tokenizer: ESMTokenizer,
    device: torch.device,
    wt_seq: str,
    csv_path: str,
    mutant_col: str = "mutant",
    score_col: str = "DMS_score",
    max_seqs: int = 500,
) -> Dict:
    """Load a ProteinGym-format CSV, score each mutant, and compute Spearman rho.

    Parameters
    ----------
    model, tokenizer, device :
        Model, tokenizer, and device for inference.
    wt_seq : str
        Wild-type sequence.
    csv_path : str
        Path to ProteinGym CSV file.
    mutant_col : str
        Column name for mutation strings (default: ``"mutant"``).
    score_col : str
        Column name for experimental DMS scores (default: ``"DMS_score"``).
    max_seqs : int
        Maximum number of sequences to evaluate (default: 500).

    Returns
    -------
    Dict
        ``{"spearman_rho": float, "n_evaluated": int, "results": List[dict]}``
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError(
            "scipy is required for Spearman correlation. Install with: pip install scipy"
        )

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_seqs:
                break

    predicted_scores = []
    experimental_scores = []
    results = []

    for i, row in enumerate(rows):
        mut_str = row[mutant_col].strip()
        try:
            exp_score = float(row[score_col])
        except (ValueError, KeyError):
            continue

        # Multi-site mutations are colon-separated, e.g. "A2T:G10S"
        try:
            mut_parts = mut_str.split(":")
            mutations = [parse_mutation(m.strip()) for m in mut_parts]
        except Exception as e:
            print(f"  [WARN] Skipping malformed mutation '{mut_str}': {e}")
            continue

        # Validate positions are within sequence
        valid = True
        for pos_0, wt_aa, mut_aa in mutations:
            if pos_0 < 0 or pos_0 >= len(wt_seq):
                print(
                    f"  [WARN] Position {pos_0+1} out of range for seq len "
                    f"{len(wt_seq)}, skipping '{mut_str}'"
                )
                valid = False
                break
        if not valid:
            continue

        pred_score = score_mutation(model, tokenizer, device, wt_seq, mutations)
        predicted_scores.append(pred_score)
        experimental_scores.append(exp_score)
        results.append({
            "mutant": mut_str,
            "predicted_score": pred_score,
            "experimental_score": exp_score,
        })

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(rows)} mutations...")

    if len(predicted_scores) < 2:
        print("[WARN] Not enough valid mutations to compute Spearman rho.")
        return {"spearman_rho": float("nan"), "n_evaluated": len(results), "results": results}

    rho, pval = spearmanr(predicted_scores, experimental_scores)
    print(f"\nSpearman rho = {rho:.4f}  (p={pval:.3e}, n={len(results)})")
    return {"spearman_rho": float(rho), "n_evaluated": len(results), "results": results}


def evaluate_builtin_gfp(
    model: torch.nn.Module,
    tokenizer: ESMTokenizer,
    device: torch.device,
) -> Dict:
    """Run the built-in GFP benchmark.

    Scores each mutation in ``GFP_MUTATIONS`` and computes Spearman rho
    against pseudo-scores derived from expected effects.

    Parameters
    ----------
    model, tokenizer, device :
        Model, tokenizer, and device for inference.

    Returns
    -------
    Dict
        ``{"spearman_rho": float, "n_evaluated": int, "results": List[dict]}``
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")

    print(f"\nRunning built-in GFP benchmark ({len(GFP_MUTATIONS)} mutations)...")
    print(f"WT sequence length: {len(GFP_WT_SEQ)} aa")

    predicted_scores = []
    expected_scores = []
    results = []

    for mut_str, expected_effect in GFP_MUTATIONS:
        try:
            pos_0, wt_aa, mut_aa = parse_mutation(mut_str)
        except Exception as e:
            print(f"  [WARN] Could not parse '{mut_str}': {e}")
            continue

        if pos_0 < 0 or pos_0 >= len(GFP_WT_SEQ):
            print(
                f"  [WARN] Position {pos_0+1} out of range "
                f"(seq len={len(GFP_WT_SEQ)})"
            )
            continue

        # Verify WT amino acid matches
        actual_wt = GFP_WT_SEQ[pos_0]
        if actual_wt != wt_aa:
            print(
                f"  [WARN] WT mismatch at pos {pos_0+1}: "
                f"expected {wt_aa}, got {actual_wt}"
            )

        pred_score = score_mutation(
            model, tokenizer, device, GFP_WT_SEQ, [(pos_0, wt_aa, mut_aa)]
        )
        exp_score = _EFFECT_SCORE[expected_effect]

        predicted_scores.append(pred_score)
        expected_scores.append(exp_score)
        results.append({
            "mutant": mut_str,
            "expected_effect": expected_effect,
            "predicted_score": pred_score,
            "expected_score": exp_score,
        })
        print(
            f"  {mut_str:8s}  expected={expected_effect:12s}  "
            f"predicted_score={pred_score:+.4f}"
        )

    if len(predicted_scores) < 2:
        return {"spearman_rho": float("nan"), "n_evaluated": len(results), "results": results}

    rho, pval = spearmanr(predicted_scores, expected_scores)
    print(f"\nSpearman rho = {rho:.4f}  (p={pval:.3e}, n={len(results)})")
    print("(Note: rho with random model is expected to be near 0)")
    return {"spearman_rho": float(rho), "n_evaluated": len(results), "results": results}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and tokenizer from a checkpoint file.

    The checkpoint should be a dict with at least ``"model_state_dict"`` and
    optionally ``"model_size"`` (defaults to ``"8M"``).

    Parameters
    ----------
    checkpoint_path : str
        Path to ``.pt`` checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    Tuple[ESMModel, ESMTokenizer]
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_size = ckpt.get("model_size", "8M")
    print(f"  Model size: {model_size}")

    tokenizer = ESMTokenizer()
    model = build_model(model_size=model_size, vocab_size=tokenizer.vocab_size)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot mutation fitness prediction using ESM-2 style model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--wt_seq", type=str, default=None,
        help="Wild-type sequence string. If not set, uses built-in avGFP WT sequence.",
    )
    parser.add_argument(
        "--dms_csv", type=str, default=None,
        help=(
            "Path to ProteinGym-format CSV file. "
            "If not set, runs built-in GFP benchmark."
        ),
    )
    parser.add_argument(
        "--mutant_col", type=str, default="mutant",
        help="Column name for mutation strings in the CSV.",
    )
    parser.add_argument(
        "--score_col", type=str, default="DMS_score",
        help="Column name for experimental DMS scores in the CSV.",
    )
    parser.add_argument(
        "--max_seqs", type=int, default=500,
        help="Maximum number of sequences to evaluate from CSV.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="If set, save per-mutation results to this CSV file.",
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_checkpoint(args.checkpoint, device)

    # Wild-type sequence
    wt_seq = args.wt_seq if args.wt_seq is not None else GFP_WT_SEQ
    print(f"WT sequence length: {len(wt_seq)} aa")

    # Evaluate
    if args.dms_csv is not None:
        results_dict = evaluate_from_csv(
            model, tokenizer, device,
            wt_seq=wt_seq,
            csv_path=args.dms_csv,
            mutant_col=args.mutant_col,
            score_col=args.score_col,
            max_seqs=args.max_seqs,
        )
    else:
        results_dict = evaluate_builtin_gfp(model, tokenizer, device)

    print(f"\n=== Summary ===")
    print(f"Spearman rho : {results_dict['spearman_rho']:.4f}")
    print(f"N evaluated  : {results_dict['n_evaluated']}")

    # Save results
    if args.output is not None:
        results = results_dict["results"]
        if results:
            fieldnames = list(results[0].keys())
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
