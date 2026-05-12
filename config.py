from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data" / "filings"
INDEX_DIR   = BASE_DIR / "index"
LOGS_DIR    = BASE_DIR / "logs"
PROMPTS_DIR = BASE_DIR / "prompts"

for _d in [DATA_DIR, INDEX_DIR, LOGS_DIR, PROMPTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding model  (local — no API key required)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
SENTENCES_PER_CHUNK = 6   # sentences grouped into one chunk
SENTENCE_OVERLAP    = 2   # sentences shared between adjacent chunks

# ---------------------------------------------------------------------------
# Chunk quality filter
# ---------------------------------------------------------------------------
# Minimum word count a chunk must have to be kept.
# Chunks below this threshold are dropped during ingestion as noise.
# Set to 0 to disable.
MIN_CHUNK_WORDS = 5

# Enable/disable the minimum-word-count filter entirely.
ENABLE_MIN_WORD_FILTER = True

# ---------------------------------------------------------------------------
# Retrieval feature flags
# Each retriever leg can be independently disabled for ablation / debugging.
# At least one of ENABLE_BM25 / ENABLE_SEMANTIC must be True.
# ---------------------------------------------------------------------------
ENABLE_BM25               = True   # BM25 keyword retrieval leg
ENABLE_SEMANTIC           = True   # FAISS semantic retrieval leg
ENABLE_METADATA_FILTERING = True   # pre-filter chunks by query-detected metadata

# ---------------------------------------------------------------------------
# Retrieval hyperparameters
# ---------------------------------------------------------------------------
TOP_K                = 10    # candidates fetched by each retriever leg
SIMILARITY_THRESHOLD = 0.90  # cosine similarity floor; below this → no answer
BM25_WEIGHT          = 0.4   # EnsembleRetriever weight for BM25 leg
FAISS_WEIGHT         = 0.6   # EnsembleRetriever weight for FAISS leg

# ---------------------------------------------------------------------------
# LLM  (Ollama, local)
# ---------------------------------------------------------------------------
OLLAMA_MODEL       = "gemma4:e4b"
OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 2048

# ---------------------------------------------------------------------------
# Index artefact filenames  (all live inside INDEX_DIR)
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = INDEX_DIR / "faiss_store"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
DOC_IDS_FILE    = INDEX_DIR / "doc_ids.json"
DOCS_FILE       = INDEX_DIR / "docs.pkl"
ENTITY_MAP_FILE = INDEX_DIR / "entity_map.json"
