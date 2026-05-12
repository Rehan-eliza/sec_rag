"""
ingest.py
---------
Responsible for:
  1. Parsing the standardised SEC filing header block
  2. Splitting the body into sentence-level chunks
  3. Attaching metadata to every chunk as a LangChain Document

Header format expected at the top of every .txt filing:
    Company: Pfizer Inc
    Ticker: PFE
    Filing Type: 10-Q (Quarterly Report)
    Filing Date: 2023-11-08
    Report Period: 2023-10-01
    Quarter: 2023Q4
    CIK: 0000078003          ← dropped
    Source: SEC EDGAR         ← dropped
    URL: https://...          ← dropped
    ============================================================

Chunk metadata fields
---------------------
  word_count   : int   — total word count of the chunk text
  bm25_tokens  : list  — pre-tokenised form (lowercase, stopwords removed)
                         stored here at ingest time so BM25 index builds
                         from the precomputed field rather than re-tokenising
                         page_content on every retriever rebuild.
"""

from __future__ import annotations

import re
import string
import uuid
from pathlib import Path

import nltk
from langchain_core.documents import Document

from config import (
    DATA_DIR,
    ENABLE_MIN_WORD_FILTER,
    MIN_CHUNK_WORDS,
    SENTENCE_OVERLAP,
    SENTENCES_PER_CHUNK,
)

# ---------------------------------------------------------------------------
# NLTK bootstrap  (downloads only if not already present)
# ---------------------------------------------------------------------------
for _pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOPWORDS       = set(nltk.corpus.stopwords.words("english"))
HEADER_SEPARATOR = re.compile(r"={10,}")   # 10+ equals signs

# Fields we keep (lowercase key → canonical field name)
_KEEP_FIELDS = {
    "company":       "company",
    "ticker":        "ticker",
    "filing type":   "filing_type",
    "filing date":   "filing_date",
    "report period": "report_period",
    "quarter":       "quarter",
}


# ---------------------------------------------------------------------------
# Shared tokeniser
# Defined here (single source of truth) and imported by retriever.py
# so that ingest-time and query-time tokenisation are always identical.
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, strip English possessive ('s), other punctuation, then stopwords."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"(?<=[a-z0-9])'s\b", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------

def parse_header(raw: str) -> tuple[dict, str]:
    """
    Split raw filing text into (metadata_dict, body_text).
    Returns an empty metadata dict if no header separator is found,
    treating the whole file as body text.
    """
    match = HEADER_SEPARATOR.search(raw)
    if not match:
        return {}, raw.strip()

    header_block = raw[: match.start()]
    body         = raw[match.end():].strip()

    metadata: dict[str, str] = {}
    for line in header_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key_norm = key.strip().lower()
        if key_norm in _KEEP_FIELDS:
            metadata[_KEEP_FIELDS[key_norm]] = value.strip()

    return metadata, body


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------

def sentence_chunk(body: str, metadata: dict, source_file: str) -> list[Document]:
    """
    Split body into overlapping sentence windows.
    Each Document inherits the filing-level metadata plus:
      - doc_id       : unique UUID string
      - source_file  : originating filename
      - chunk_index  : position within this filing
      - word_count   : number of whitespace-separated words in the chunk
      - bm25_tokens  : pre-tokenised list (lowercase, no stopwords)

    Chunks below MIN_CHUNK_WORDS are dropped when ENABLE_MIN_WORD_FILTER
    is True in config.
    """
    sentences = nltk.sent_tokenize(body)
    # Drop trivially short sentences (page numbers, lone headers, etc.)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 4]

    if not sentences:
        return []

    chunks: list[Document] = []
    step = max(1, SENTENCES_PER_CHUNK - SENTENCE_OVERLAP)

    for start in range(0, len(sentences), step):
        window = sentences[start: start + SENTENCES_PER_CHUNK]
        if not window:
            break

        chunk_text  = " ".join(window)
        word_count  = len(chunk_text.split())
        bm25_tokens = tokenize(chunk_text)

        # --- Minimum word filter ---
        if ENABLE_MIN_WORD_FILTER and word_count < MIN_CHUNK_WORDS:
            continue

        doc_id = str(uuid.uuid4())
        doc = Document(
            page_content=chunk_text,
            metadata={
                **metadata,
                "doc_id":      doc_id,
                "source_file": source_file,
                "chunk_index": len(chunks),
                "word_count":  word_count,
                "bm25_tokens": bm25_tokens,
            },
        )
        chunks.append(doc)

    return chunks


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def load_filing_text(raw: str, source_name: str = "unknown") -> list[Document]:
    """Parse a single filing string into a list of chunked Documents."""
    metadata, body = parse_header(raw)
    if not body:
        return []
    return sentence_chunk(body, metadata, source_name)


def load_filing_file(path: Path) -> list[Document]:
    """Load a .txt filing from disk."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return load_filing_text(raw, source_name=path.name)


def load_all_filings(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every .txt file in data_dir and return all chunks."""
    all_docs: list[Document] = []
    txt_files = sorted(data_dir.glob("*.txt"))

    if not txt_files:
        print(f"[ingest] No .txt files found in {data_dir}")
        return all_docs

    for fp in txt_files:
        docs = load_filing_file(fp)
        print(f"[ingest] {fp.name}: {len(docs)} chunks")
        all_docs.extend(docs)

    print(f"[ingest] Total chunks: {len(all_docs)}")
    return all_docs
