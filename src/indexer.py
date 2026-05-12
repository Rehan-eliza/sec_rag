"""
indexer.py
----------
Builds and persists:
  - FAISS vector store  (via LangChain + sentence-transformers)
  - Raw embeddings array  (numpy, normalised — dot product == cosine similarity)
  - Doc-id list  (parallel to embeddings array for O(1) lookup)
  - Pickled Document list  (full chunk store with metadata)
  - Entity map  (ticker / company-name → metadata values, for query parsing)

Run once before launching the app:
    python build_index.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    DOCS_FILE,
    DOC_IDS_FILE,
    EMBEDDINGS_FILE,
    ENTITY_MAP_FILE,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL,
)


# ---------------------------------------------------------------------------
# Embedding model  (singleton — loaded once per process)
# ---------------------------------------------------------------------------

_embeddings_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        print(f"[indexer] Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},  # dot product == cosine sim
        )
    return _embeddings_model


# ---------------------------------------------------------------------------
# Entity map builder
# ---------------------------------------------------------------------------

def build_entity_map(docs: list[Document]) -> dict[str, dict]:
    """
    Build a lowercase lookup dict from every unique ticker and company name
    found in the corpus.

    Example output:
        {
          "tsla": {"ticker": "TSLA", "company": "Tesla, Inc."},
          "tesla": {"ticker": "TSLA", "company": "Tesla, Inc."},
          "tesla, inc.": {"ticker": "TSLA", "company": "Tesla, Inc."},
          ...
        }
    """
    entity_map: dict[str, dict] = {}

    for doc in docs:
        meta    = doc.metadata
        ticker  = meta.get("ticker", "").strip()
        company = meta.get("company", "").strip()

        if not ticker:
            continue

        entry = {"ticker": ticker, "company": company}

        # Index by ticker (e.g. "tsla")
        if ticker:
            entity_map[ticker.lower()] = entry

        # Index by full company name (e.g. "tesla, inc.")
        if company:
            entity_map[company.lower()] = entry

        # Index by first word of company name (e.g. "tesla")
        if company:
            first_word = company.lower().split()[0].rstrip(",.")
            entity_map[first_word] = entry

    return entity_map


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(docs: list[Document]) -> None:
    """
    Embed all documents, build FAISS store, and persist all artefacts to disk.
    """
    if not docs:
        raise ValueError("[indexer] No documents provided — cannot build index.")

    model = get_embedding_model()

    print(f"[indexer] Embedding {len(docs)} chunks …")
    texts = [d.page_content for d in docs]

    # Raw embeddings (normalised — for cosine similarity via dot product)
    raw_embeddings: list[list[float]] = model.embed_documents(texts)
    embeddings_array = np.array(raw_embeddings, dtype=np.float32)

    # FAISS vector store via LangChain
    print("[indexer] Building FAISS index …")
    faiss_store = FAISS.from_documents(docs, model)
    faiss_store.save_local(str(FAISS_INDEX_DIR))
    print(f"[indexer] FAISS saved → {FAISS_INDEX_DIR}")

    # Embeddings array + doc_ids (parallel arrays for O(1) lookup)
    doc_ids = [d.metadata["doc_id"] for d in docs]
    np.save(str(EMBEDDINGS_FILE), embeddings_array)
    DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2))
    print(f"[indexer] Embeddings saved → {EMBEDDINGS_FILE}")

    # Full document list (chunk text + metadata)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    print(f"[indexer] Chunk store saved → {DOCS_FILE}")

    # Entity map
    entity_map = build_entity_map(docs)
    ENTITY_MAP_FILE.write_text(json.dumps(entity_map, indent=2))
    print(f"[indexer] Entity map saved → {ENTITY_MAP_FILE} ({len(entity_map)} entries)")

    print("[indexer] ✓ Index build complete.")


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------

def load_index() -> tuple[FAISS, list[Document], dict[str, np.ndarray], dict]:
    """
    Load all persisted artefacts from disk.

    Returns:
        faiss_store     — LangChain FAISS vectorstore
        all_docs        — list of all Document chunks
        embeddings_map  — {doc_id: normalised_embedding_vector}
        entity_map      — {lowercase_name_or_ticker: {ticker, company}}
    """
    missing = [
        p for p in [FAISS_INDEX_DIR, EMBEDDINGS_FILE, DOC_IDS_FILE, DOCS_FILE, ENTITY_MAP_FILE]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"[indexer] Index artefacts missing: {missing}\n"
            "Run  python build_index.py  first."
        )

    model = get_embedding_model()

    faiss_store = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        model,
        allow_dangerous_deserialization=True,
    )

    embeddings_array = np.load(str(EMBEDDINGS_FILE))
    doc_ids          = json.loads(DOC_IDS_FILE.read_text())
    embeddings_map   = {doc_id: embeddings_array[i] for i, doc_id in enumerate(doc_ids)}

    with open(DOCS_FILE, "rb") as f:
        all_docs: list[Document] = pickle.load(f)

    entity_map = json.loads(ENTITY_MAP_FILE.read_text())

    print(f"[indexer] Index loaded — {len(all_docs)} chunks, {len(entity_map)} entity entries")
    return faiss_store, all_docs, embeddings_map, entity_map


def index_exists() -> bool:
    return all(
        p.exists()
        for p in [FAISS_INDEX_DIR, EMBEDDINGS_FILE, DOC_IDS_FILE, DOCS_FILE, ENTITY_MAP_FILE]
    )
