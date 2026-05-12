"""
inspect_chunks.py
-----------------
Visualises the word-count distribution of all indexed chunks so you can
make an informed decision about MIN_CHUNK_WORDS in config.py.

Usage
-----
    python inspect_chunks.py              # uses persisted index
    python inspect_chunks.py --rebuild    # re-ingests from data/filings/

Output
------
  - Console: percentile table + summary stats
  - Saves:   logs/chunk_distribution.png

Reading the chart
-----------------
  - The histogram shows chunk frequency by word count.
  - The red dashed line marks the current MIN_CHUNK_WORDS threshold.
  - The percentile table helps you decide: e.g. if p5 = 12 words, setting
    MIN_CHUNK_WORDS=12 would drop the noisiest 5% of chunks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from config import LOGS_DIR, MIN_CHUNK_WORDS, ENABLE_MIN_WORD_FILTER


# ---------------------------------------------------------------------------
# Attempt matplotlib import — optional dependency
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[inspect] matplotlib not installed — console output only.")
    print("[inspect] Install with:  pip install matplotlib")


# ---------------------------------------------------------------------------
# Load or build docs
# ---------------------------------------------------------------------------

def get_docs(rebuild: bool = False):
    from src.indexer import index_exists, load_index
    from src.ingest  import load_all_filings
    from config      import DATA_DIR

    if rebuild or not index_exists():
        print("[inspect] Ingesting from data/filings/ …")
        docs = load_all_filings(DATA_DIR)
        if not docs:
            print("[inspect] No documents found. Add .txt files to data/filings/")
            sys.exit(1)
        return docs
    else:
        print("[inspect] Loading from persisted index …")
        _, docs, _, _ = load_index()
        return docs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(docs) -> np.ndarray:
    """Extract word counts from all chunks."""
    counts = []
    for doc in docs:
        # Use stored word_count if present, otherwise compute
        wc = doc.metadata.get("word_count")
        if wc is None:
            wc = len(doc.page_content.split())
        counts.append(wc)
    return np.array(counts, dtype=np.int32)


def print_report(counts: np.ndarray) -> None:
    """Print percentile table and summary stats to console."""
    percentiles = [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
    pct_values  = np.percentile(counts, percentiles).astype(int)

    print("\n" + "=" * 52)
    print(f"  Chunk word-count distribution  ({len(counts):,} chunks)")
    print("=" * 52)
    print(f"  {'Statistic':<20}  {'Words':>8}")
    print("  " + "-" * 30)
    print(f"  {'Min':<20}  {int(counts.min()):>8}")
    print(f"  {'Mean':<20}  {int(counts.mean()):>8}")
    print(f"  {'Median (p50)':<20}  {int(np.median(counts)):>8}")
    print(f"  {'Max':<20}  {int(counts.max()):>8}")
    print(f"  {'Std dev':<20}  {int(counts.std()):>8}")
    print()
    print(f"  {'Percentile':<20}  {'Words':>8}")
    print("  " + "-" * 30)
    for p, v in zip(percentiles, pct_values):
        marker = " ← current threshold" if ENABLE_MIN_WORD_FILTER and v == MIN_CHUNK_WORDS else ""
        print(f"  {f'p{p}':<20}  {v:>8}{marker}")

    print()
    if ENABLE_MIN_WORD_FILTER:
        n_dropped = int((counts < MIN_CHUNK_WORDS).sum())
        pct_drop  = 100 * n_dropped / len(counts)
        print(f"  Current MIN_CHUNK_WORDS = {MIN_CHUNK_WORDS}")
        print(f"  Chunks that would be dropped: {n_dropped:,} ({pct_drop:.1f}%)")
    else:
        print("  ENABLE_MIN_WORD_FILTER = False (no chunks dropped)")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_distribution(counts: np.ndarray, out_path: Path) -> None:
    if not _HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Chunk Word-Count Distribution  ({len(counts):,} chunks)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ---- Left: histogram (full range) ----
    ax = axes[0]
    ax.set_title("Full distribution", fontsize=11)
    n, bins, patches = ax.hist(
        counts, bins=60, color="#3C7DD9", edgecolor="white", linewidth=0.4, alpha=0.85
    )
    ax.set_xlabel("Word count per chunk", fontsize=10)
    ax.set_ylabel("Number of chunks", fontsize=10)

    if ENABLE_MIN_WORD_FILTER:
        ax.axvline(
            MIN_CHUNK_WORDS, color="#D94F3C", linestyle="--", linewidth=1.5,
            label=f"MIN_CHUNK_WORDS = {MIN_CHUNK_WORDS}",
        )
        ax.legend(fontsize=9)

    # Percentile annotations
    for p, col in [(25, "#27AE60"), (50, "#F39C12"), (75, "#8E44AD")]:
        val = int(np.percentile(counts, p))
        ax.axvline(val, color=col, linestyle=":", linewidth=1.2, alpha=0.7,
                   label=f"p{p} = {val}")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)

    # ---- Right: zoomed-in tail (≤ p95) ----
    ax2 = axes[1]
    p95 = int(np.percentile(counts, 95))
    zoomed = counts[counts <= p95]
    ax2.set_title(f"Zoomed — up to p95 ({p95} words)", fontsize=11)
    ax2.hist(
        zoomed, bins=50, color="#27AE60", edgecolor="white", linewidth=0.4, alpha=0.85
    )
    ax2.set_xlabel("Word count per chunk", fontsize=10)
    ax2.set_ylabel("Number of chunks", fontsize=10)

    if ENABLE_MIN_WORD_FILTER and MIN_CHUNK_WORDS <= p95:
        ax2.axvline(
            MIN_CHUNK_WORDS, color="#D94F3C", linestyle="--", linewidth=1.5,
            label=f"MIN_CHUNK_WORDS = {MIN_CHUNK_WORDS}",
        )
        ax2.legend(fontsize=9)

    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.grid(axis="y", alpha=0.3)

    # ---- Percentile table inset (right plot) ----
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals    = [int(np.percentile(counts, p)) for p in percentiles]
    table_text  = "\n".join(f"p{p:>2}: {v:>5}" for p, v in zip(percentiles, pct_vals))
    ax2.text(
        0.97, 0.97, table_text,
        transform=ax2.transAxes, fontsize=7.5,
        verticalalignment="top", horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[inspect] Chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect chunk word-count distribution.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-ingest from data/filings/ instead of loading persisted index."
    )
    args = parser.parse_args()

    docs   = get_docs(rebuild=args.rebuild)
    counts = analyse(docs)

    print_report(counts)

    out_path = LOGS_DIR / "chunk_distribution.png"
    plot_distribution(counts, out_path)


if __name__ == "__main__":
    main()
