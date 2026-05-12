"""
build_index.py
--------------
One-time script to ingest all filings in data/filings/ and build the
FAISS + BM25 index artifacts in index/.

Run this before launching the app:
    python build_index.py

Options
-------
    --data-dir PATH   Override the default data/filings/ directory.
    --force           Rebuild even if index artifacts already exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from src.indexer import build_index, index_exists
from src.ingest import load_all_filings


def main():
    parser = argparse.ArgumentParser(description="Build the SEC filing index.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists.")
    args = parser.parse_args()

    if index_exists() and not args.force:
        print("[build_index] Index already exists. Use --force to rebuild.")
        sys.exit(0)

    print(f"[build_index] Ingesting from: {args.data_dir}")
    docs = load_all_filings(args.data_dir)

    if not docs:
        print("[build_index] No documents found. Exiting.")
        sys.exit(1)

    build_index(docs)
    print("[build_index] Done.")


if __name__ == "__main__":
    main()
