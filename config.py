from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env from the project root — must happen before any os.getenv calls
load_dotenv(Path(__file__).parent / ".env")

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
# Hugging Face token
# Set HF_TOKEN in your .env file — never put it in this file directly.
# .env is listed in .gitignore and will not be committed to source control.
# ---------------------------------------------------------------------------
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Embedding model  (local — no external API calls)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Batch size for encoding at index build time.
EMBEDDING_BATCH_SIZE = 1024

# ---------------------------------------------------------------------------
# Chunking
#
# The chunker uses a SLIDING WINDOW over sentences, not one-sentence-per-chunk.
# SENTENCES_PER_CHUNK controls how many sentences are grouped into one chunk.
# SENTENCE_OVERLAP controls how many sentences are shared between adjacent
# chunks (provides context continuity across chunk boundaries).
#
# Example with SENTENCES_PER_CHUNK=3, SENTENCE_OVERLAP=1:
#   Chunk 1: sentences 1, 2, 3
#   Chunk 2: sentences 3, 4, 5   <- sentence 3 repeated for continuity
#   Chunk 3: sentences 5, 6, 7
#
# Set SENTENCES_PER_CHUNK=1 and SENTENCE_OVERLAP=0 for true single-sentence
# chunking.
# ---------------------------------------------------------------------------
SENTENCES_PER_CHUNK = 6   # sentences grouped into one chunk window
SENTENCE_OVERLAP    = 2   # sentences shared between adjacent windows

# ---------------------------------------------------------------------------
# Chunk quality filter
# ---------------------------------------------------------------------------
MIN_CHUNK_WORDS        = 5     # drop chunks below this word count
ENABLE_MIN_WORD_FILTER = True  # set False to disable the filter entirely

# ---------------------------------------------------------------------------
# Retrieval feature flags
# ---------------------------------------------------------------------------
ENABLE_BM25               = True
ENABLE_SEMANTIC           = True
ENABLE_METADATA_FILTERING = True

# ---------------------------------------------------------------------------
# Retrieval hyperparameters
# ---------------------------------------------------------------------------
TOP_K                = 10
# Cosine floor for FAISS candidates only (before RRF). BM25 uses top-k only.
SIMILARITY_THRESHOLD = 0.90
BM25_WEIGHT          = 0.5
FAISS_WEIGHT         = 0.5

# ---------------------------------------------------------------------------
# LLM  (Ollama, local)
# ---------------------------------------------------------------------------
OLLAMA_MODEL       = "gemma4:e4b"
OLLAMA_HOST        = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 2048

# ---------------------------------------------------------------------------
# Index artifact filenames
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = INDEX_DIR / "faiss_store"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
DOC_IDS_FILE    = INDEX_DIR / "doc_ids.json"
DOCS_FILE       = INDEX_DIR / "docs.pkl"
ENTITY_MAP_FILE = INDEX_DIR / "entity_map.json"