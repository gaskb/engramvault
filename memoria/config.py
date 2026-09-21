import os
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.getenv("MEMORIA_DATA_DIR", Path.home() / ".memoria"))
DEFAULT_DB_PATH = Path(os.getenv("MEMORIA_DB_PATH", DEFAULT_DATA_DIR / "vault.db"))

# LLM Configuration
LLM_PROVIDER = os.getenv("MEMORIA_LLM_PROVIDER", "")
LLM_BASE_URL = os.getenv("MEMORIA_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("MEMORIA_LLM_API_KEY", "dummy")
LLM_MODEL = os.getenv("MEMORIA_LLM_MODEL", "qwen2.5:3b")

# Chunking settings
CHUNK_SIZE = int(os.getenv("MEMORIA_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("MEMORIA_CHUNK_OVERLAP", "150"))

# Search weights for RRF
RRF_K = int(os.getenv("MEMORIA_RRF_K", "60"))
BM25_WEIGHT = float(os.getenv("MEMORIA_BM25_WEIGHT", "1.0"))
VEC_WEIGHT = float(os.getenv("MEMORIA_VEC_WEIGHT", "1.0"))


def ensure_data_dir() -> Path:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR
