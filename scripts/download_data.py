"""
download_data.py
================
Download Swiss-Prot reviewed protein sequences from UniProt REST API,
deduplicate by sequence, filter by length, and split into train/val sets.

Usage
-----
python download_data.py --output_dir ./data

# Custom split and length filters:
python download_data.py --output_dir ./data --split_ratio 0.95 \\
    --min_len 10 --max_len 512 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# UniProt API
# ---------------------------------------------------------------------------

UNIPROT_API_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=fasta&query=reviewed:true"
)


# ---------------------------------------------------------------------------
# FASTA I/O helpers
# ---------------------------------------------------------------------------


def parse_fasta_stream(lines) -> List[Tuple[str, str]]:
    """Parse FASTA lines into (header, sequence) pairs.

    Parameters
    ----------
    lines : iterable of str
        Lines from a FASTA file or HTTP response.

    Returns
    -------
    List[Tuple[str, str]]
        List of ``(header, sequence)`` pairs.
    """
    records = []
    current_header = None
    current_seq_parts: List[str] = []

    for line in lines:
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_seq_parts)))
            current_header = line[1:]
            current_seq_parts = []
        else:
            current_seq_parts.append(line.upper())

    if current_header is not None:
        records.append((current_header, "".join(current_seq_parts)))

    return records


def write_fasta(records: List[Tuple[str, str]], path: str, line_width: int = 60) -> None:
    """Write (header, sequence) pairs to a FASTA file.

    Parameters
    ----------
    records : List[Tuple[str, str]]
        List of ``(header, sequence)`` pairs.
    path : str
        Output file path.
    line_width : int
        Characters per sequence line (default: 60).
    """
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_swissprot(
    output_fasta: Optional[str] = None,
    chunk_size: int = 1024 * 1024,
    timeout: int = 300,
) -> List[Tuple[str, str]]:
    """Download Swiss-Prot reviewed sequences from UniProt REST API.

    Parameters
    ----------
    output_fasta : str, optional
        If set, also save the raw downloaded FASTA to this path.
    chunk_size : int
        Download chunk size in bytes (default: 1 MB).
    timeout : int
        HTTP request timeout in seconds (default: 300).

    Returns
    -------
    List[Tuple[str, str]]
        List of ``(header, sequence)`` pairs.
    """
    print(f"Downloading Swiss-Prot from UniProt REST API...")
    print(f"  URL: {UNIPROT_API_URL}")

    req = urllib.request.Request(
        UNIPROT_API_URL,
        headers={"User-Agent": "download_data.py/1.0 (protein-plm-lab)"},
    )

    lines_buffer: List[str] = []
    bytes_downloaded = 0
    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            partial_line = ""
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                bytes_downloaded += len(chunk)
                text = partial_line + chunk.decode("utf-8", errors="replace")
                split_lines = text.split("\n")
                # Last element may be incomplete
                partial_line = split_lines[-1]
                lines_buffer.extend(split_lines[:-1])

                elapsed = time.time() - start_time
                mb = bytes_downloaded / 1024 / 1024
                print(f"\r  Downloaded {mb:.1f} MB in {elapsed:.0f}s...", end="", flush=True)

            if partial_line:
                lines_buffer.append(partial_line)

    except urllib.error.URLError as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    mb = bytes_downloaded / 1024 / 1024
    print(f"\n  Download complete: {mb:.1f} MB in {elapsed:.1f}s")

    if output_fasta is not None:
        with open(output_fasta, "w") as f:
            f.write("\n".join(lines_buffer))
        print(f"  Raw FASTA saved to: {output_fasta}")

    records = parse_fasta_stream(lines_buffer)
    print(f"  Parsed {len(records):,} records")
    return records


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------


def deduplicate(records: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate sequences (keep first occurrence).

    Deduplication is by exact sequence match (case-insensitive).

    Parameters
    ----------
    records : List[Tuple[str, str]]
        Input records.

    Returns
    -------
    List[Tuple[str, str]]
        Deduplicated records.
    """
    seen: set = set()
    unique = []
    for header, seq in records:
        seq_upper = seq.upper()
        seq_hash = hashlib.md5(seq_upper.encode()).hexdigest()
        if seq_hash not in seen:
            seen.add(seq_hash)
            unique.append((header, seq_upper))
    n_removed = len(records) - len(unique)
    print(f"  Deduplication: {len(records):,} -> {len(unique):,} (removed {n_removed:,} duplicates)")
    return unique


def filter_by_length(
    records: List[Tuple[str, str]],
    min_len: int = 10,
    max_len: int = 512,
) -> List[Tuple[str, str]]:
    """Filter sequences by length.

    Parameters
    ----------
    records : List[Tuple[str, str]]
        Input records.
    min_len : int
        Minimum sequence length (inclusive, default: 10).
    max_len : int
        Maximum sequence length (inclusive, default: 512).

    Returns
    -------
    List[Tuple[str, str]]
        Filtered records.
    """
    filtered = [(h, s) for h, s in records if min_len <= len(s) <= max_len]
    n_removed = len(records) - len(filtered)
    print(
        f"  Length filter [{min_len}, {max_len}]: "
        f"{len(records):,} -> {len(filtered):,} (removed {n_removed:,})"
    )
    return filtered


def train_val_split(
    records: List[Tuple[str, str]],
    split_ratio: float = 0.95,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Randomly split records into train and validation sets.

    Parameters
    ----------
    records : List[Tuple[str, str]]
        Input records.
    split_ratio : float
        Fraction of records for training (default: 0.95).
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        ``(train_records, val_records)``
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * split_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train:]
    print(f"  Train/val split ({split_ratio:.0%}/{1-split_ratio:.0%}): "
          f"{len(train):,} train, {len(val):,} val")
    return train, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Swiss-Prot reviewed sequences from UniProt, "
            "deduplicate, filter by length, and split into train/val."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Directory to save output FASTA files.",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.95,
        help="Fraction of sequences for training (rest goes to validation).",
    )
    parser.add_argument(
        "--max_len", type=int, default=512,
        help="Maximum sequence length to keep.",
    )
    parser.add_argument(
        "--min_len", type=int, default=10,
        help="Minimum sequence length to keep.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Download
    raw_fasta_path = os.path.join(args.output_dir, "swissprot_raw.fasta")
    records = download_swissprot(output_fasta=raw_fasta_path)

    print(f"\nProcessing {len(records):,} records...")

    # Deduplicate
    records = deduplicate(records)

    # Filter by length
    records = filter_by_length(records, min_len=args.min_len, max_len=args.max_len)

    # Train/val split
    train_records, val_records = train_val_split(
        records, split_ratio=args.split_ratio, seed=args.seed
    )

    # Save
    train_path = os.path.join(args.output_dir, "swissprot_train.fasta")
    val_path = os.path.join(args.output_dir, "swissprot_val.fasta")

    write_fasta(train_records, train_path)
    write_fasta(val_records, val_path)

    print(f"\n=== Done ===")
    print(f"Train FASTA : {train_path}  ({len(train_records):,} sequences)")
    print(f"Val FASTA   : {val_path}  ({len(val_records):,} sequences)")


if __name__ == "__main__":
    main()
