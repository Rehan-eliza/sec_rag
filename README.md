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
    → year/filing-type regex
    → metadata pre-filter narrows chunk pool
    → BM25 top-10  +  FAISS top-10  (run in parallel)
    → EnsembleRetriever (LangChain RRF fusion) → merged top-10
    → cosine similarity threshold (≥ 90%)
    → if no chunks qualify → warning, no answer
    → prompt assembly → single Ollama call (gemma4:e4b)
    → Streamlit UI with inline citations + similarity scores
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
| `BM25_WEIGHT` | `0.4` | RRF weight for BM25 leg |
| `FAISS_WEIGHT` | `0.6` | RRF weight for FAISS leg |
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
├── index/                  # Auto-generated index artefacts
├── logs/                   # inspect_chunks.py output
├── prompts/
│   └── prompt_log.md       # Prompt iteration history
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
- **Local-only** — no external API calls. Embeddings via `sentence-transformers`,
  LLM via Ollama, BM25 via `rank-bm25`.
- **Metadata filtering before retrieval** — narrows the chunk pool before BM25
  and FAISS run, improving both precision and speed.
- **90% cosine similarity threshold** — conservative by design. If the corpus
  does not contain relevant information the system says so rather than
  hallucinating.
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
- Review `prompts/prompt_log.md` for the reasoning behind the current
  prompt template.