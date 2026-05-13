"""
app.py
------
Streamlit front-end for the SEC RAG system.
Matches the UI design agreed during architecture review:
  - Status pills (sources matched, min similarity, model)
  - Answer with inline [N] citation badges
  - Per-source cards: similarity bar, metadata pills, verbatim excerpt
  - Warning state when retrieval returns no chunks
  - Sidebar: index status, file uploader, rebuild index
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    ENABLE_BM25,
    ENABLE_SEMANTIC,
    OLLAMA_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from src.indexer import build_index, get_embedding_model, index_exists, load_index
from src.ingest import load_all_filings, load_filing_text
from src.llm import model_is_available, ollama_is_available, stream_ollama
from src.prompt import build_citation_map, build_prompt
from src.retriever import retrieve

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEC Filing Analyst",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — matches the design system from the mockup
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Pill badges */
.pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 500; margin-right: 6px;
}
.pill-green  { background: #EAF3DE; color: #27500A; }
.pill-teal   { background: #E1F5EE; color: #085041; }
.pill-gray   { background: #F1EFE8; color: #444441; }
.pill-purple { background: #EEEDFE; color: #3C3489; }
.pill-red    { background: #FCEBEB; color: #791F1F; }
.pill-amber  { background: #FAEEDA; color: #633806; }

/* Source card */
.source-card {
    border: 0.5px solid #e0ded6;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: #ffffff;
}
/* Similarity bar */
.sim-bar-bg {
    height: 6px; background: #f0ede6;
    border-radius: 3px; width: 100%;
}
.sim-bar-fill {
    height: 6px; border-radius: 3px; background: #639922;
}
/* Monospace excerpt */
.excerpt {
    font-family: monospace; font-size: 12px;
    color: #5f5e5a; line-height: 1.6;
    margin-top: 8px;
}
/* Warning box */
.warn-box {
    border: 0.5px solid #f7c1c1;
    border-radius: 12px; padding: 16px 20px;
    background: #fcebeb; color: #791f1f;
    margin-top: 12px;
}
/* Answer prose */
.answer-prose {
    font-size: 15px; line-height: 1.75;
    color: #2c2c2a;
}
/* Citation superscript badge */
sup .cite-badge {
    display: inline-block;
    padding: 1px 5px; border-radius: 10px;
    font-size: 10px; font-weight: 500;
    color: white; vertical-align: super;
    line-height: 1;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Citation badge colours (cycle for up to 10 sources)
# ---------------------------------------------------------------------------
_BADGE_COLORS = [
    "#085041",  # teal
    "#3C3489",  # purple
    "#633806",  # amber
    "#4A1B0C",  # coral
    "#27500A",  # green
    "#185FA5",  # blue
    "#791F1F",  # red
    "#72243E",  # pink
    "#5F5E5A",  # gray
    "#3B6D11",  # green-2
]


def _render_answer(answer: str, n_sources: int) -> str:
    """Replace [N] markers with coloured superscript badges."""
    for i in range(1, n_sources + 1):
        color = _BADGE_COLORS[(i - 1) % len(_BADGE_COLORS)]
        badge = (
            f'<sup><span class="cite-badge" style="background:{color}">'
            f"{i}</span></sup>"
        )
        answer = re.sub(rf"\[{i}\]", badge, answer)
    return answer


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
if "embeddings_map" not in st.session_state:
    st.session_state.embeddings_map = {}
if "entity_map" not in st.session_state:
    st.session_state.entity_map = {}
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "extra_docs" not in st.session_state:
    st.session_state.extra_docs = []   # docs from uploaded files (session only)
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


# ---------------------------------------------------------------------------
# Index loader helper
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index…")
def _cached_load_index():
    return load_index()


def ensure_index_loaded():
    if not st.session_state.index_loaded:
        if index_exists():
            (
                st.session_state.faiss_store,
                st.session_state.all_docs,
                st.session_state.embeddings_map,
                st.session_state.entity_map,
            ) = _cached_load_index()
            st.session_state.index_loaded = True


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("SEC RAG")
    st.caption("Retrieval-augmented analysis of SEC filings")
    st.divider()

    # --- System status ---
    st.subheader("System status")
    ollama_ok = ollama_is_available()
    model_ok  = model_is_available() if ollama_ok else False
    idx_ok    = index_exists()

    st.markdown(
        f"{'🟢' if ollama_ok else '🔴'} Ollama server  \n"
        f"{'🟢' if model_ok  else '🔴'} Model `{OLLAMA_MODEL}`  \n"
        f"{'🟢' if idx_ok   else '🔴'} Filing index"
    )

    if not ollama_ok:
        st.warning("Start Ollama with `ollama serve`", icon="⚠️")
    if ollama_ok and not model_ok:
        st.warning(f"Pull the model: `ollama pull {OLLAMA_MODEL}`", icon="⚠️")

    st.divider()

    # --- Index management ---
    st.subheader("Index")
    if idx_ok:
        n_docs = len(st.session_state.all_docs) or "?"
        st.success(f"Index ready ({n_docs} chunks)")
    else:
        st.info("No index found. Build it below.")

    if st.button("🔨 Build / Rebuild index", use_container_width=True):
        with st.spinner("Building index from data/filings/ …"):
            docs = load_all_filings(DATA_DIR)
            if docs:
                build_index(docs)
                # Clear cache so next run reloads
                _cached_load_index.clear()
                st.session_state.index_loaded = False
                st.session_state.extra_docs   = []
                st.success(f"Index built — {len(docs)} chunks.")
                st.rerun()
            else:
                st.error("No .txt files found in data/filings/")

    st.divider()

    # --- File uploader ---
    st.subheader("Upload filings")
    st.caption("Uploaded files are added to this session's retrieval pool.")
    uploaded = st.file_uploader(
        "Drop SEC filing .txt files here",
        type=["txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        new_docs = []
        for uf in uploaded:
            raw = uf.read().decode("utf-8", errors="replace")
            new_docs.extend(load_filing_text(raw, source_name=uf.name))

        if new_docs:
            # Merge with session extra docs (deduplicate by source_file)
            existing_names = {d.metadata.get("source_file") for d in st.session_state.extra_docs}
            added = [d for d in new_docs if d.metadata.get("source_file") not in existing_names]
            st.session_state.extra_docs.extend(added)
            if added:
                st.success(f"+{len(added)} chunks from uploaded file(s).")

    if st.session_state.extra_docs:
        st.caption(f"Session pool: {len(st.session_state.extra_docs)} extra chunks")

    st.divider()

    # --- Config info ---
    st.subheader("Config")
    _cfg_lines = [
        f"**Model:** {OLLAMA_MODEL}",
        f"**Top-K:** {TOP_K}",
    ]
    if ENABLE_SEMANTIC:
        _cfg_lines.append(
            f"**FAISS (semantic leg):** cosine ≥ {SIMILARITY_THRESHOLD:.0%} before RRF"
        )
    if ENABLE_BM25:
        _cfg_lines.append(f"**BM25:** top-{TOP_K} only (no cosine threshold)")
    if ENABLE_BM25 and ENABLE_SEMANTIC:
        _cfg_lines.append("**Fusion:** weighted RRF")
    st.caption("  \n".join(_cfg_lines))

# ---------------------------------------------------------------------------
# Load index on app start
# ---------------------------------------------------------------------------
ensure_index_loaded()

# Merge uploaded session docs into the retrieval pool
combined_docs = list(st.session_state.all_docs) + list(st.session_state.extra_docs)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("SEC Filing Analyst")
st.caption(
    "Ask a business question about the SEC filings corpus. "
    "Every claim in the answer is grounded in a specific filing."
)

# Example questions (write into query_input so Analyze rerun keeps the text)
with st.expander("Example questions"):
    examples = [
        "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
        "How has NVIDIA's revenue and growth outlook changed over the last two years?",
        "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"example_btn_{i}", use_container_width=True):
            st.session_state.query_input = ex

if "prefill_query" in st.session_state:
    st.session_state.query_input = st.session_state.pop("prefill_query")

# Query input — key binds widget to session_state so value survives Analyze reruns
query = st.text_area(
    "Your question",
    key="query_input",
    height=90,
    placeholder="e.g. What are the primary risk factors facing Apple and Tesla?",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    if st.session_state.query_history:
        if st.button("Clear history", use_container_width=False):
            st.session_state.query_history = []
            st.rerun()

# Shown only during an Analyze run (after gates pass) so clicks feel responsive
_analysis_progress_slot = st.empty()

# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------
if submitted:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not st.session_state.index_loaded and not combined_docs:
        st.error("No index loaded and no filings in session. Build the index or upload files.")
        st.stop()

    if not ollama_ok:
        st.error("Ollama is not running. Start it with `ollama serve`.")
        st.stop()

    if not model_ok:
        st.error(f"Model `{OLLAMA_MODEL}` is not available. Run `ollama pull {OLLAMA_MODEL}`.")
        st.stop()

    progress = _analysis_progress_slot.progress(
        0,
        text="Retrieving relevant filing excerpts from the corpus…",
    )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------
    t0 = time.time()

    all_docs_pool = combined_docs
    faiss_store   = st.session_state.faiss_store
    embeddings_map = st.session_state.embeddings_map
    entity_map    = st.session_state.entity_map

    # Merge extra_docs embeddings if uploaded files were indexed
    # (uploaded docs go through retriever's BM25 only unless re-indexed;
    #  they are included in the BM25 path and cosine sim is best-effort)
    if not all_docs_pool:
        _analysis_progress_slot.empty()
        st.error("No documents in the retrieval pool.")
        st.stop()

    qualified, filters, rrf_counts = retrieve(
        query=query,
        all_docs=all_docs_pool,
        faiss_store=faiss_store,
        embeddings_map=embeddings_map,
        embedding_model=get_embedding_model(),
        entity_map=entity_map,
    )
    retrieval_time = time.time() - t0

    progress.progress(
        0.35,
        text=(
            f"Retrieval done ({retrieval_time:.1f}s) — "
            f"{len(qualified)} excerpt(s) passed filters; preparing the prompt…"
        ),
    )

    # -----------------------------------------------------------------------
    # No-results warning
    # -----------------------------------------------------------------------
    if not qualified:
        _analysis_progress_slot.empty()
        filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items()) if filters else "none"
        st.markdown(
            f"""
<div class="warn-box">
  <strong>No answer — insufficient evidence</strong><br>
  No filing excerpts were retrieved for your query.<br>
  <small>Filters applied: {filter_desc}</small><br><br>
  <em>Try broadening your question, or check that the relevant filings are in the corpus.</em>
</div>
""",
            unsafe_allow_html=True,
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Prompt assembly
    # -----------------------------------------------------------------------
    prompt       = build_prompt(query, qualified)
    citation_map = build_citation_map(qualified)

    progress.progress(
        0.55,
        text=(
            f"Calling `{OLLAMA_MODEL}` — generating the answer "
            "(first tokens can take a few seconds)…"
        ),
    )

    # -----------------------------------------------------------------------
    # LLM call (stream tokens so the UI updates as the model generates)
    # -----------------------------------------------------------------------
    t1 = time.time()
    try:
        streamed = st.write_stream(stream_ollama(prompt))
    except RuntimeError as e:
        _analysis_progress_slot.empty()
        st.error(str(e))
        st.stop()
    llm_time = time.time() - t1
    raw_answer = (streamed or "").strip()

    progress.progress(
        1.0,
        text=f"Answer complete ({llm_time:.1f}s) — saving to history…",
    )

    # -----------------------------------------------------------------------
    # Store in history
    # -----------------------------------------------------------------------
    result = {
        "query":        query,
        "answer":       raw_answer,
        "citation_map": citation_map,
        "filters":      filters,
        "retrieval_ms": int(retrieval_time * 1000),
        "llm_ms":       int(llm_time * 1000),
        "rrf_counts":   rrf_counts,
    }
    st.session_state.query_history.insert(0, result)
    _analysis_progress_slot.empty()
    st.rerun()

# ---------------------------------------------------------------------------
# Render results  (most-recent first)
# ---------------------------------------------------------------------------
for result in st.session_state.query_history:
    cmap = result["citation_map"]

    # --- Question echo ---
    st.markdown(
        f"<p style='font-size:13px;color:#888780;margin:0 0 4px'>Question</p>"
        f"<p style='font-size:16px;font-weight:500;margin:0 0 14px'>"
        f"{result['query']}</p>",
        unsafe_allow_html=True,
    )

    # --- Status pills ---
    n        = len(cmap)
    min_sim  = min(v["similarity"] for v in cmap.values()) if cmap else 0
    pills_html = (
        f'<span class="pill pill-green">✓ {n} source{"s" if n != 1 else ""} matched</span>'
        f'<span class="pill pill-teal">Min similarity {min_sim:.0%}</span>'
        f'<span class="pill pill-purple">{OLLAMA_MODEL}</span>'
        f'<span class="pill pill-gray">'
        f'Retrieval {result["retrieval_ms"]}ms · LLM {result["llm_ms"]}ms'
        f"</span>"
    )
    if result.get("rrf_counts"):
        rc = result["rrf_counts"]
        pills_html += (
            f'<span class="pill pill-gray">'
            f'RRF · BM25 {rc["bm25"]} · Semantic {rc["semantic"]}'
            f"</span>"
        )

    if result["filters"]:
        filter_str = " · ".join(
            (", ".join(v) if isinstance(v, list) else str(v))
            for v in result["filters"].values()
        )
        pills_html += f'<span class="pill pill-amber">Filtered: {filter_str}</span>'

    st.markdown(pills_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Answer with inline citations ---
    rendered_answer = _render_answer(result["answer"], n)
    st.markdown(
        f'<div class="answer-prose">{rendered_answer}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Sources ---
    st.markdown(
        "<p style='font-size:13px;font-weight:500;color:#888780;margin:0 0 8px'>"
        "Sources</p>",
        unsafe_allow_html=True,
    )

    for idx, info in cmap.items():
        sim_pct   = int(info["similarity"] * 100)
        bar_width = sim_pct
        badge_col = _BADGE_COLORS[(idx - 1) % len(_BADGE_COLORS)]

        pills = (
            f'<span class="pill" style="background:{badge_col}20;color:{badge_col}">'
            f"{idx}</span>"
            f'<span class="pill pill-gray">{info["ticker"]}</span>'
            f'<span class="pill pill-gray">{info["filing_type"]}</span>'
            f'<span class="pill pill-gray">{info["period"]}</span>'
        )
        if info.get("source_file"):
            pills += (
                f'<span class="pill pill-gray">'
                f'{info["source_file"]}</span>'
            )

        excerpt_escaped = (
            info["excerpt"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

        card_html = f"""
<div class="source-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
    <div style="flex:1">
      <div style="margin-bottom:6px">{pills}</div>
      <div class="excerpt">"{excerpt_escaped}"</div>
    </div>
    <div style="min-width:68px;text-align:right">
      <div style="font-size:13px;font-weight:500;color:#27500A">{sim_pct}%</div>
      <div class="sim-bar-bg" style="margin:4px 0">
        <div class="sim-bar-fill" style="width:{bar_width}%"></div>
      </div>
      <div style="font-size:11px;color:#888780">similarity</div>
    </div>
  </div>
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)

    # --- Reliability footnote ---
    st.markdown(
        f"""
<div style="background:#f1efe8;border-radius:8px;padding:10px 14px;margin-top:4px">
  <small style="color:#5f5e5a">
  The FAISS leg uses cosine ≥ {SIMILARITY_THRESHOLD:.0%} to the query before RRF; BM25 uses top-{TOP_K} only.
  The bar is cosine vs your question (not an RRF score). All quotes are verbatim extracts from SEC filings.
  </small>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()
