# SEC Filing RAG System

Retrieval-augmented generation over SEC EDGAR filings (10-K / 10-Q).
Answers natural-language business questions in a single LLM call,
with every claim traced to a specific filing excerpt.

---

## Architecture

```
.txt filings
    → header parser (rules-based metadata extraction)
    → sentence chunker (word_count + bm25_tokens stored per chunk)
    → FAISS vector index  +  BM25Okapi index  (persisted to index/)

Query
    → entity detector (dict lookup → ticker/company)
    → year + filing-type hints (regex + keywords)
    → metadata pre-filter narrows chunk pool (optional)
    → BM25 top-K  +  FAISS top-K  (both legs when enabled; run in parallel)
    → cosine floor on FAISS candidates only (BM25 leg is top-K, no cosine gate)
    → weighted reciprocal-rank fusion of the two leg lists (LangChain helper)
    → fused list cosine-scored for citation ordering in the UI
    → if nothing survives filters / semantic-only gate → warning, no answer
    → prompt assembly → single Ollama call (default: gemma4:e4b)
    → Streamlit UI: streamed answer, inline citations + similarity scores
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- `gemma4:e4b` model pulled in Ollama

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull the model

```bash
ollama pull gemma4:e4b
```

### 3. Add filings

Place SEC filing `.txt` files in `data/filings/`.  
Each file must start with the standard header block:

```
Company: Pfizer Inc
Ticker: PFE
Filing Type: 10-Q (Quarterly Report)
Filing Date: 2023-11-08
Report Period: 2023-10-01
Quarter: 2023Q4
CIK: 0000078003
Source: SEC EDGAR
URL: https://www.sec.gov/...
============================================================
... filing body text ...
```

### 4. Build the index

```bash
python build_index.py
```

Use `--force` to rebuild an existing index:

```bash
python build_index.py --force
```

### 5. Launch the app

```bash
# In a separate terminal, make sure Ollama is running:
ollama serve

# Then:
streamlit run app.py
```

---

## Tuning chunk quality

Inspect the word-count distribution of your chunks before running queries:

```bash
python inspect_chunks.py
```

This prints a percentile table to the console and saves a chart to
`logs/chunk_distribution.png`. Use it to set `MIN_CHUNK_WORDS` in
`config.py`.

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `SENTENCES_PER_CHUNK` | `6` | Sentences per chunk window |
| `SENTENCE_OVERLAP` | `2` | Overlap between adjacent chunks |
| `MIN_CHUNK_WORDS` | `5` | Drop chunks below this word count |
| `ENABLE_MIN_WORD_FILTER` | `True` | Toggle the word-count filter |
| `ENABLE_BM25` | `True` | Toggle BM25 retrieval leg |
| `ENABLE_SEMANTIC` | `True` | Toggle FAISS semantic retrieval leg |
| `ENABLE_METADATA_FILTERING` | `True` | Toggle metadata pre-filtering |
| `TOP_K` | `10` | Candidates per retriever leg |
| `SIMILARITY_THRESHOLD` | `0.90` | Cosine similarity floor |
| `BM25_WEIGHT` | `0.5` | RRF weight for BM25 leg |
| `FAISS_WEIGHT` | `0.5` | RRF weight for FAISS leg |
| `OLLAMA_MODEL` | `gemma4:e4b` | Model served by Ollama |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM temperature |

---

## Project structure

```
sec-rag/
├── app.py                  # Streamlit front-end
├── build_index.py          # One-time index build script
├── inspect_chunks.py       # Chunk distribution visualisation
├── config.py               # All tuneable settings
├── requirements.txt
├── data/
│   └── filings/            # Place .txt filing files here
├── index/                  # Auto-generated index artifacts
├── logs/                   # inspect_chunks.py output
├── prompts/                # Created at runtime (optional notes / prompt history)
└── src/
    ├── ingest.py            # Header parsing + sentence chunking
    ├── indexer.py           # FAISS build/load + entity map
    ├── retriever.py         # BM25 + FAISS + RRF pipeline
    ├── prompt.py            # Prompt template + assembler
    └── llm.py               # Single Ollama call
```

---

## Example questions

```
What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?
How has NVIDIA's revenue and growth outlook changed over the last two years?
What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?
```

---

## Design decisions & assumptions

- **Single LLM call constraint** — all retrieval, filtering, and prompt assembly
  happens in Python. The model receives one fully-assembled prompt and produces
  one response.
- **No cloud LLM API** — answers come from a local Ollama model; embeddings use
  `sentence-transformers` (first model download may hit Hugging Face; optional
  `HF_TOKEN` in `.env` for gated models). BM25 via `rank-bm25`.
- **Metadata filtering before retrieval** — narrows the chunk pool before BM25
  and FAISS run, improving both precision and speed.
- **90% cosine floor on the semantic leg** — FAISS candidates below the
  threshold are dropped before fusion; BM25 hits are not cosine-filtered. When
  semantic-only mode filters everything, the UI warns instead of answering.
- **Pre-computed `bm25_tokens`** — stored on each chunk at ingest time so the
  BM25 tokenisation is guaranteed consistent between index and query time.
- **`word_count` metadata field** — stored per chunk to enable the distribution
  inspection script and the configurable minimum-word filter without re-parsing
  chunk text at retrieval time.

---

## Evaluating quality

- Run `inspect_chunks.py` to verify chunk size distribution is sensible.
- Use the similarity score shown per source in the UI — low scores on all
  sources indicate the corpus doesn't cover the question well.
- Toggle `ENABLE_BM25` / `ENABLE_SEMANTIC` off independently to observe
  the contribution of each retrieval leg.
- Read the header comments in `src/prompt.py` for template versioning; keep
  optional notes under `prompts/` if you maintain a separate prompt log.

---

## To do

- **Ingest / metadata** — Support PDFs and layout-aware extractors (e.g. Azure Document Intelligence) so chunk metadata is richer and retrieval can filter and match more precisely.
- **Retrieval** — Rerank to a smaller top‑K per leg before RRF; apply **MMR** (maximal marginal relevance) after reranking and before fusion so legs contribute less redundant overlap.
- **Embeddings / index** — Swap or add a domain-specific financial embedding model so the semantic (FAISS) leg matches filing language better.
- **Chunking** — Experiment with parent–child chunks, light graph structure across related passages, or an improved sliding window to reduce boundary loss.
- **Generation** — Offer a frontier-class model option for higher-quality synthesis when extraction from sources is subtle or dense.
- **Infra** — Scale CPU/GPU so index build and embedding passes finish faster on large corpora.
- **Storage / retrieval** — Move off pure in-memory FAISS to something like Postgres or MongoDB (with cold storage tiers) for reuse across runs; enable heavier pre-filtering (e.g. k-means on chunk embeddings) before full vector search.
- **Feedback loop** — Persist user ratings, final answers, and the chunks supplied; optionally short-circuit new queries that closely match highly rated past Q&A pairs and reuse the validated answer instead of re-running the full pipeline.
- **Query prep** — Query rewriting and/or **HyDE** (hypothetical document embeddings) so short or vague questions become retrieval-friendly text before the semantic leg runs.
- **Query routing** — Classify each query with rules or a small model (fact lookup, comparison, risk narrative, …) and branch retrieval: e.g. stricter metadata filters for single-ticker facts, higher `TOP_K` or per-ticker retrieval for comparisons, broader recall when answers span long qualitative sections—instead of one fixed setting for every question.
- **Multi-agent orchestration** — Add a **master agent** that inspects each query and routes work to **sub-agents** (e.g. dedicated retrieval, table/numbers, cross-filing comparison, final answer synthesis) so complex questions run targeted tools and prompts instead of one monolithic pipeline path.
- **Retrieval (calibration)** — Learn RRF leg weights and similarity thresholds from labeled `(query, relevant chunk IDs)` pairs instead of fixed constants.
- **Section tags** — Tag chunks with filing sections (MD&A, Risk Factors, Notes to financial statements, etc.) from layout or headings so filters and boosting match where answers usually live.
- **Tables / chunking** — Preserve table rows and layout-adjacent structure so numeric questions retain columns and headers in context.
- **Generation / consistency** — Use chain-of-thought where it helps break down complex questions; add consistency checks on model output (e.g. **UQLM** or similar packages) before surfacing answers.
- **Evaluation** — Build a curated labeled set to tune chunking, retrieval, fusion, and prompts with measurable regressions.
- **CI/CD & prompts** — Automate pipeline checks against that eval set; integrate user feedback into dynamic prompt variants; adopt a small agent only if it clearly improves orchestration over static prompts.