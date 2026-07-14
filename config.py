from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
RESULTS_DIR = DATA_DIR / "results"

DEFAULT_MODEL_CARDS_FILE = RAW_DATA_DIR / "model_cards.jsonl"
DEFAULT_DOCUMENTS_FILE = PROCESSED_DATA_DIR / "documents.jsonl"
DEFAULT_INDEX_DIR = INDEX_DIR / "default"

DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DIM = 384
N_VECTORS = 100_000
N_QUERIES = 100

HNSW_M = 32
IVF_NLIST = 100
IVF_NPROBE = 10
IVF_M = 8
IVF_NBITS = 8