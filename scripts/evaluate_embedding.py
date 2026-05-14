"""
evaluate_embedding.py
=====================
Evaluate embedding quality of a trained ESM-2 model using k-NN classification.
Sequences are grouped by taxonomy (bacteria/archaea/eukaryota/virus) based on
UniProt FASTA headers, and CLS token embeddings are used as features.

Usage
-----
python evaluate_embedding.py \\
    --checkpoint /path/to/checkpoint.pt \\
    --fasta /path/to/swissprot.fasta \\
    --n_seqs 1000 --k 5 --pooling cls

# With t-SNE plot:
python evaluate_embedding.py \\
    --checkpoint /path/to/checkpoint.pt \\
    --fasta /path/to/swissprot.fasta \\
    --tsne --output results.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow importing from /workspace/esm2/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import build_model
from tokenizer import ESMTokenizer


# ---------------------------------------------------------------------------
# Taxonomy extraction
# ---------------------------------------------------------------------------


def extract_taxonomy(header: str) -> str:
    """Extract taxonomy group from a UniProt FASTA header.

    UniProt header format::

        >sp|P12345|GENE_HUMAN Protein name OS=Homo sapiens OX=9606 GN=... PE=... SV=...

    Classification rules (checked against the OS= field, case-insensitive):

    * ``"virus"``, ``"phage"``, ``"bacteriophage"`` -> ``"virus"``
    * ``"bacteria"``, ``"bacillus"``, ``"escherichia"``, ``"salmonella"``,
      ``"staphylococcus"``, ``"streptococcus"``, ``"mycobacterium"``,
      ``"pseudomonas"``, ``"clostridium"`` -> ``"bacteria"``
    * ``"archaea"``, ``"archaeon"``, ``"methanobacterium"``, ``"sulfolobus"``,
      ``"halobacterium"`` -> ``"archaea"``
    * everything else -> ``"eukaryota"``

    Parameters
    ----------
    header : str
        FASTA header line (with or without leading ``>``).

    Returns
    -------
    str
        One of ``"bacteria"``, ``"archaea"``, ``"eukaryota"``, ``"virus"``.
    """
    # Extract OS= field value (everything up to the next two-letter tag or end)
    os_match = re.search(r"OS=(.+?)(?:\s+[A-Z]{2}=|$)", header)
    os_str = os_match.group(1).lower() if os_match else header.lower()

    # Virus keywords
    virus_keywords = ["virus", "phage", "bacteriophage", "viridae", "virales"]
    for kw in virus_keywords:
        if kw in os_str:
            return "virus"

    # Bacteria keywords
    bacteria_keywords = [
        "bacteria", "bacterial", "bacillus", "escherichia", "salmonella",
        "staphylococcus", "streptococcus", "mycobacterium", "pseudomonas",
        "clostridium", "lactobacillus", "listeria", "helicobacter",
        "campylobacter", "vibrio", "klebsiella", "enterococcus",
    ]
    for kw in bacteria_keywords:
        if kw in os_str:
            return "bacteria"

    # Archaea keywords
    archaea_keywords = [
        "archaea", "archaeon", "archaeal", "methanobacterium", "sulfolobus",
        "halobacterium", "methanococcus", "thermococcus", "pyrococcus",
        "archaeoglobus", "methanopyrus",
    ]
    for kw in archaea_keywords:
        if kw in os_str:
            return "archaea"

    # Default: eukaryota
    return "eukaryota"


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------


def parse_fasta(fasta_path: str, max_seqs: int = 1000) -> Tuple[List[str], List[str]]:
    """Parse a FASTA file and return (headers, sequences).

    Parameters
    ----------
    fasta_path : str
        Path to FASTA file.
    max_seqs : int
        Maximum number of sequences to load.

    Returns
    -------
    Tuple[List[str], List[str]]
        ``(headers, sequences)``
    """
    headers = []
    sequences = []
    current_header = None
    current_seq_parts: List[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append("".join(current_seq_parts))
                    if len(sequences) >= max_seqs:
                        break
                current_header = line[1:]  # strip leading >
                headers.append(current_header)
                current_seq_parts = []
            else:
                current_seq_parts.append(line.upper())

        # Don't forget the last sequence
        if current_header is not None and len(sequences) < max_seqs:
            sequences.append("".join(current_seq_parts))

    # Trim headers to match sequences count
    headers = headers[: len(sequences)]
    return headers, sequences


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def get_embeddings(
    model: torch.nn.Module,
    tokenizer: ESMTokenizer,
    device: torch.device,
    sequences: List[str],
    batch_size: int = 16,
    max_len: int = 512,
    pooling: str = "cls",
) -> np.ndarray:
    """Extract embeddings for a list of sequences.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ESM-2 style model.
    tokenizer : ESMTokenizer
        Tokenizer aligned with the model vocabulary.
    device : torch.device
        Device to run inference on.
    sequences : List[str]
        List of amino-acid sequences.
    batch_size : int
        Number of sequences per forward pass (default: 16).
    max_len : int
        Maximum sequence length; longer sequences are truncated (default: 512).
    pooling : str
        ``"cls"`` uses position 0 (CLS token) embedding;
        ``"mean"`` averages over non-special token positions (default: ``"cls"``).

    Returns
    -------
    np.ndarray
        Shape ``(N, d_model)``.
    """
    model.eval()
    all_embeddings = []

    for batch_start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[batch_start : batch_start + batch_size]

        # Truncate sequences to max_len
        batch_seqs = [s[:max_len] for s in batch_seqs]

        # Tokenize and pad
        encoded = [tokenizer.encode(s) for s in batch_seqs]
        max_batch_len = max(len(e) for e in encoded)

        padded = []
        attention_masks = []
        for enc in encoded:
            pad_len = max_batch_len - len(enc)
            padded.append(enc + [tokenizer.pad_token_id] * pad_len)
            attention_masks.append([1] * len(enc) + [0] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad():
            # model returns logits; we need hidden states
            # Use model's embed + transformer layers directly if available,
            # otherwise hook into the forward pass
            hidden = _get_hidden_states(model, input_ids)  # (B, L, d_model)

        if pooling == "cls":
            # CLS token is at position 0
            batch_emb = hidden[:, 0, :].cpu().numpy()
        else:
            # Mean pooling over non-special (non-pad, non-cls, non-eos) positions
            batch_emb_list = []
            for i, enc in enumerate(encoded):
                # positions 1 .. len(enc)-2 are amino acid tokens (skip cls and eos)
                seq_len = len(enc)
                if seq_len > 2:
                    aa_hidden = hidden[i, 1 : seq_len - 1, :]  # (L_aa, d_model)
                    batch_emb_list.append(aa_hidden.mean(dim=0).cpu().numpy())
                else:
                    batch_emb_list.append(hidden[i, 0, :].cpu().numpy())
            batch_emb = np.stack(batch_emb_list, axis=0)

        all_embeddings.append(batch_emb)

        if (batch_start // batch_size + 1) % 10 == 0:
            n_done = min(batch_start + batch_size, len(sequences))
            print(f"  Embedded {n_done}/{len(sequences)} sequences...")

    return np.concatenate(all_embeddings, axis=0)


def _get_hidden_states(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Extract last-layer hidden states from the model.

    Tries to call ``model.get_hidden_states()`` if available; otherwise
    hooks into the last transformer layer to capture its output.

    Parameters
    ----------
    model : torch.nn.Module
        ESM-2 style model.
    input_ids : torch.Tensor
        Shape ``(B, L)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, L, d_model)``.
    """
    # If the model exposes a get_hidden_states method, use it
    if hasattr(model, "get_hidden_states"):
        return model.get_hidden_states(input_ids)

    # Otherwise, register a forward hook on the last transformer layer
    hidden_states = {}

    def hook_fn(module, input, output):
        # output may be a tensor or tuple; take the first element
        if isinstance(output, tuple):
            hidden_states["last"] = output[0]
        else:
            hidden_states["last"] = output

    # Find the last transformer layer
    hook_handle = None
    last_layer = None

    # Try common attribute names for the transformer layers
    for attr in ["layers", "encoder", "transformer"]:
        layers = getattr(model, attr, None)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            last_layer = layers[-1]
            break

    if last_layer is not None:
        hook_handle = last_layer.register_forward_hook(hook_fn)

    # Run forward pass (returns logits)
    logits = model(input_ids)  # (B, L, vocab_size)

    if hook_handle is not None:
        hook_handle.remove()

    if "last" in hidden_states:
        return hidden_states["last"]

    # Fallback: use the embedding layer output by running only the embedding
    # This is a last resort; quality will be lower
    if hasattr(model, "embedding"):
        with torch.no_grad():
            return model.embedding(input_ids)

    # Ultimate fallback: derive from logits via pseudo-inverse (not ideal)
    # Just return a zero tensor of the right shape as a placeholder
    B, L, V = logits.shape
    d_model = getattr(model, "d_model", 320)
    return torch.zeros(B, L, d_model, device=input_ids.device)


# ---------------------------------------------------------------------------
# k-NN classification
# ---------------------------------------------------------------------------


def knn_accuracy(
    embeddings: np.ndarray,
    labels: List[str],
    k: int = 5,
    n_splits: int = 5,
) -> Dict:
    """k-NN classification with stratified k-fold cross-validation.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, d_model)``.
    labels : List[str]
        Class labels for each embedding.
    k : int
        Number of nearest neighbours (default: 5).
    n_splits : int
        Number of stratified k-fold splits (default: 5).

    Returns
    -------
    Dict
        ``{"mean_accuracy": float, "std_accuracy": float, "per_class": dict}``
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )

    labels_arr = np.array(labels)
    le = LabelEncoder()
    y = le.fit_transform(labels_arr)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    all_true = []
    all_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, y)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        all_true.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    # Per-class accuracy from all folds combined
    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)
    per_class = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = all_true_arr == cls_idx
        if mask.sum() > 0:
            cls_acc = float((all_pred_arr[mask] == cls_idx).mean())
            per_class[cls_name] = cls_acc

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# t-SNE visualisation
# ---------------------------------------------------------------------------


def plot_tsne(
    embeddings: np.ndarray,
    labels: List[str],
    output_path: str,
) -> None:
    """Generate and save a t-SNE scatter plot coloured by taxonomy label.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, d_model)``.
    labels : List[str]
        Taxonomy labels for each point.
    output_path : str
        Path to save the PNG figure.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "scikit-learn and matplotlib are required for t-SNE. "
            "Install with: pip install scikit-learn matplotlib"
        )

    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[label_to_color[lbl]],
            label=lbl, alpha=0.6, s=10,
        )
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title("t-SNE of protein embeddings (coloured by taxonomy)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and tokenizer from a checkpoint file.

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
        description="Evaluate ESM-2 embedding quality via k-NN taxonomy classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to Swiss-Prot FASTA file for embedding evaluation.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--n_seqs", type=int, default=1000,
        help="Maximum number of sequences to evaluate.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of nearest neighbours for k-NN classification.",
    )
    parser.add_argument(
        "--pooling", type=str, default="cls", choices=["cls", "mean"],
        help="Pooling strategy: 'cls' (CLS token) or 'mean' (mean over AA positions).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="If set, save per-sequence results to this CSV file.",
    )
    parser.add_argument(
        "--tsne", action="store_true", default=False,
        help="If set, generate and save a t-SNE scatter plot.",
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

    # Parse FASTA
    print(f"\nParsing FASTA: {args.fasta}")
    headers, sequences = parse_fasta(args.fasta, max_seqs=args.n_seqs)
    print(f"  Loaded {len(sequences)} sequences")

    # Extract taxonomy labels
    labels = [extract_taxonomy(h) for h in headers]
    from collections import Counter
    label_counts = Counter(labels)
    print(f"  Taxonomy distribution: {dict(label_counts)}")

    # Filter out classes with too few samples for cross-validation
    min_samples = args.k + 1
    valid_classes = {cls for cls, cnt in label_counts.items() if cnt >= min_samples}
    if len(valid_classes) < len(label_counts):
        removed = set(label_counts) - valid_classes
        print(f"  [WARN] Removing classes with < {min_samples} samples: {removed}")
        keep = [i for i, l in enumerate(labels) if l in valid_classes]
        headers = [headers[i] for i in keep]
        sequences = [sequences[i] for i in keep]
        labels = [labels[i] for i in keep]
        print(f"  Remaining: {len(sequences)} sequences")

    if len(sequences) == 0:
        print("[ERROR] No sequences remaining after filtering. Exiting.")
        return

    # Extract embeddings
    print(f"\nExtracting embeddings (pooling={args.pooling}, batch_size={args.batch_size})...")
    embeddings = get_embeddings(
        model, tokenizer, device,
        sequences=sequences,
        batch_size=args.batch_size,
        pooling=args.pooling,
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # k-NN accuracy
    n_splits = min(5, min(Counter(labels).values()))
    n_splits = max(2, n_splits)
    print(f"\nRunning {n_splits}-fold stratified k-NN (k={args.k})...")
    knn_results = knn_accuracy(embeddings, labels, k=args.k, n_splits=n_splits)

    print(f"\n=== Results ===")
    print(f"Mean accuracy : {knn_results['mean_accuracy']:.4f} ± {knn_results['std_accuracy']:.4f}")
    print(f"Per-class accuracy:")
    for cls_name, cls_acc in sorted(knn_results["per_class"].items()):
        print(f"  {cls_name:15s}: {cls_acc:.4f}")

    # t-SNE plot
    if args.tsne:
        tsne_path = (args.output.replace(".csv", "_tsne.png") if args.output
                     else "embedding_tsne.png")
        plot_tsne(embeddings, labels, tsne_path)

    # Save results
    if args.output is not None:
        import csv
        rows = [
            {
                "header": h,
                "taxonomy": l,
                "seq_len": len(s),
            }
            for h, l, s in zip(headers, labels, sequences)
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["header", "taxonomy", "seq_len"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-sequence info saved to: {args.output}")


if __name__ == "__main__":
    main()
