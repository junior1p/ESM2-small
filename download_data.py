#!/usr/bin/env python3
"""Download Swiss-Prot protein sequences from UniProt REST API."""

import os, requests

URL = "https://rest.uniprot.org/uniprotkb/stream"
PARAMS = {
    "query": "reviewed:true",
    "format": "fasta",
    "size": 100000,
}
OUT = "data/swissprot.fasta"

def main():
    os.makedirs("data", exist_ok=True)
    print(f"Downloading Swiss-Prot (up to 100K sequences)...")

    with requests.get(URL, params=PARAMS, stream=True, timeout=120) as r:
        total = 0
        with open(OUT, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
                if total % (10 * 1024 * 1024) < 65536:
                    print(f"  {total / 1024 / 1024:.1f} MB downloaded...")
        print(f"Done: {total / 1024 / 1024:.1f} MB → {OUT}")

if __name__ == "__main__":
    main()
