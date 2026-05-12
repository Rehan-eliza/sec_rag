"""
retriever.py
------------
Full retrieval pipeline, in order:

  1. Parse metadata filters from the user query
       - Entity detector  : dict-lookup against the corpus entity map
       - Year / quarter   : regex
       - Filing type      : keyword list
  2. Pre-filter the chunk list by those filters      (if ENABLE_METADATA_FILTERING)
  3. BM25Retriever over filtered chunks → top 10     (if ENABLE_BM25)
  4. FAISS retriever with metadata filter → top 10   (if ENABLE_SEMANTIC)
  5. LangChain EnsembleRetriever (RRF fusion)        (when both legs active)
     — or single-leg results when only one is enabled
  6. Score each result with true cosine similarity
  7. Drop any chunk below SIMILARITY_THRESHOLD
  8. Return qualified chunks (or empty list → UI shows warning)

Feature flags (config.py):
  ENABLE_BM25               — toggle BM25 keyword retrieval leg
  ENABLE_SEMANTIC           — toggle FAISS semantic retrieval leg
  ENABLE_METADATA_FILTERING — toggle metadata pre-filtering

BM25 uses the pre-computed `bm25_tokens` field stored on each Document
at ingest time (same tokenise function, guaranteed consistency).
"""

from __future__ import annotations

import re
import string
from typing import NamedTuple

import numpy as np
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import (
    BM25_WEIGHT,
    ENABLE_BM25,
    ENABLE_METADATA_FILTERING,
    ENABLE_SEMANTIC,
    FAISS_WEIGHT,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

# Import the shared tokeniser from ingest — single source of truth.
# This guarantees that BM25 index-time and query-time tokenisation
# are always identical.
from src.ingest import tokenize  # noqa: E402


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ScoredChunk(NamedTuple):
    document:   Document
    similarity: float   # cosine similarity ∈ [0, 1]


# ---------------------------------------------------------------------------
# Query filter extraction
# ---------------------------------------------------------------------------

_YEAR_RE      = re.compile(r"\b(20\d{2})\b")
_ANNUAL_KW    = {"annual", "10-k", "10k", "yearly", "year-end"}
_QUARTERLY_KW = {"quarterly", "10-q", "10q", "quarter"}


def parse_query_filters(query: str, entity_map: dict) -> dict:
    """
    Return a dict of metadata filters inferred from the query.
    Keys that may appear: tickers (list), filing_type (str), year (str).
    All keys are optional — only present when evidence is found.
    """
    filters: dict = {}
    lower = query.lower()
    words = lower.translate(str.maketrans("", "", string.punctuation)).split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    # --- Company / ticker detection ---
    tickers: list[str] = []
    for token in words + bigrams:
        if token in entity_map:
            ticker = entity_map[token]["ticker"]
            if ticker not in tickers:
                tickers.append(ticker)
    if tickers:
        filters["tickers"] = tickers

    # --- Year ---
    year_match = _YEAR_RE.search(query)
    if year_match:
        filters["year"] = year_match.group(1)

    # --- Filing type ---
    words_set = set(words)
    if words_set & _ANNUAL_KW:
        filters["filing_type"] = "10-K"
    elif words_set & _QUARTERLY_KW:
        filters["filing_type"] = "10-Q"

    return filters


# ---------------------------------------------------------------------------
# Metadata pre-filter
# ---------------------------------------------------------------------------

def filter_docs(all_docs: list[Document], filters: dict) -> list[Document]:
    """
    Narrow the full chunk list using extracted metadata filters.
    Returns all docs unchanged if ENABLE_METADATA_FILTERING is False
    or if no filters were detected.
    """
    if not ENABLE_METADATA_FILTERING or not filters:
        return all_docs

    result = []
    for doc in all_docs:
        meta = doc.metadata

        if "tickers" in filters:
            if meta.get("ticker") not in filters["tickers"]:
                continue

        if "year" in filters:
            yr = filters["year"]
            filing_year  = (meta.get("filing_date")  or "")[:4]
            report_year  = (meta.get("report_period") or "")[:4]
            quarter_year = (meta.get("quarter")       or "")[:4]
            if yr not in (filing_year, report_year, quarter_year):
                continue

        if "filing_type" in filters:
            ft = meta.get("filing_type", "")
            if filters["filing_type"] not in ft:
                continue

        result.append(doc)

    return result


# ---------------------------------------------------------------------------
# FAISS metadata filter callable
# ---------------------------------------------------------------------------

def make_faiss_filter(filters: dict):
    """Return a callable for FAISS search_kwargs['filter'], or None."""
    if not ENABLE_METADATA_FILTERING or not filters:
        return None

    def _filter(meta: dict) -> bool:
        if "tickers" in filters:
            if meta.get("ticker") not in filters["tickers"]:
                return False
        if "year" in filters:
            yr = filters["year"]
            if not any(
                (meta.get(k) or "")[:4] == yr
                for k in ("filing_date", "report_period", "quarter")
            ):
                return False
        if "filing_type" in filters:
            if filters["filing_type"] not in (meta.get("filing_type") or ""):
                return False
        return True

    return _filter


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    all_docs: list[Document],
    faiss_store: FAISS,
    embeddings_map: dict[str, np.ndarray],
    embedding_model,
    entity_map: dict,
) -> tuple[list[ScoredChunk], dict]:
    """
    Run the full retrieval pipeline.

    Returns
    -------
    qualified : list[ScoredChunk]
        Chunks that passed the cosine similarity threshold, sorted descending.
        Empty list → caller should surface the no-answer warning.
    filters   : dict
        The metadata filters that were applied (for display in the UI).
    """
    if not ENABLE_BM25 and not ENABLE_SEMANTIC:
        raise ValueError("At least one of ENABLE_BM25 or ENABLE_SEMANTIC must be True in config.")

    # 1. Parse filters
    filters = parse_query_filters(query, entity_map)

    # 2. Pre-filter chunk list
    filtered_docs = filter_docs(all_docs, filters)
    if not filtered_docs:
        return [], filters

    # 3 & 4. Build active retriever legs
    retrievers = []
    weights    = []

    if ENABLE_BM25:
        # Use pre-computed bm25_tokens if available, else tokenise on the fly
        def _get_tokens(doc: Document) -> list[str]:
            stored = doc.metadata.get("bm25_tokens")
            return stored if stored else tokenize(doc.page_content)

        corpus_tokens = [_get_tokens(d) for d in filtered_docs]
        bm25_retriever = BM25Retriever(
            vectorizer=BM25Okapi(corpus_tokens),
            docs=list(filtered_docs),
            preprocess_func=tokenize,
            k=TOP_K,
        )

        retrievers.append(bm25_retriever)
        weights.append(BM25_WEIGHT)

    if ENABLE_SEMANTIC:
        faiss_filter      = make_faiss_filter(filters)
        faiss_search_kwargs: dict = {"k": TOP_K}
        if faiss_filter is not None:
            faiss_search_kwargs["filter"] = faiss_filter

        faiss_retriever = faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs=faiss_search_kwargs,
        )
        retrievers.append(faiss_retriever)
        weights.append(FAISS_WEIGHT)

    # 5. Fuse results
    if len(retrievers) == 1:
        fused_docs: list[Document] = retrievers[0].invoke(query)
    else:
        ensemble    = EnsembleRetriever(retrievers=retrievers, weights=weights)
        fused_docs  = ensemble.invoke(query)

    # 6. Compute true cosine similarity
    query_embedding = np.array(
        embedding_model.embed_query(query), dtype=np.float32
    )
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm

    scored: list[ScoredChunk] = []
    for doc in fused_docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id and doc_id in embeddings_map:
            doc_emb = embeddings_map[doc_id]  # normalised at index time
            sim     = float(np.dot(query_embedding, doc_emb))
            scored.append(ScoredChunk(document=doc, similarity=sim))

    # 7. Threshold + sort
    qualified = [s for s in scored if s.similarity >= SIMILARITY_THRESHOLD]
    qualified.sort(key=lambda s: s.similarity, reverse=True)

    return qualified, filters
