import os
from pathlib import Path

# Versioning (Release Version + API Contract Version + MCP Protocol Version)
APP_VERSION = "2.5.1"
UI_VERSION = "2.5.1"
API_VERSION = "v1"
MCP_PROTOCOL_VERSION = "2.0"
APP_TITLE = os.getenv("MEMORIA_APP_TITLE", "Engram AI")
APP_SUBTITLE = os.getenv("MEMORIA_APP_SUBTITLE", "AI Memory Vault")

# Security & Authentication (Issue #15)
API_TOKEN = os.getenv("MEMORIA_API_TOKEN", "").strip()
WEB_PASSWORD = os.getenv("MEMORIA_WEB_PASSWORD", "").strip()
DEFAULT_THEME_PALETTE = os.getenv("MEMORIA_DEFAULT_THEME", "emerald").lower()

# Default Language & Instance White-Labeling (Issues #13 & #16)
DEFAULT_LANG = os.getenv("MEMORIA_DEFAULT_LANG", "en").lower()
INSTANCE_NAME = os.getenv("MEMORIA_INSTANCE_NAME", "Local Node")
DOCS_URL = os.getenv("MEMORIA_DOCS_URL", "")
DOCS_LABEL = os.getenv("MEMORIA_DOCS_LABEL", "Documentation")
REPO_URL = os.getenv("MEMORIA_REPO_URL", "https://github.com/gaskb/memoria-ai")
REPO_LABEL = os.getenv("MEMORIA_REPO_LABEL", "Repository")
DESIGN_SYSTEM_URL = os.getenv("MEMORIA_DESIGN_SYSTEM_URL", "")

# Embedding Backend Configuration (Issue #14: 'auto', 'api', 'fastembed', 'native')
EMBEDDING_BACKEND = os.getenv("MEMORIA_EMBEDDING_BACKEND", "auto").lower()
EMBEDDING_MODEL = os.getenv("MEMORIA_EMBEDDING_MODEL", "")

# Optional System 1 Fast Decision Engine (Issue #18: TypeSafe Jev / OpenRouter)
JEV_ENABLED = os.getenv("MEMORIA_JEV_ENABLED", "false").lower() in ("true", "1", "yes")
JEV_MODEL = os.getenv("MEMORIA_JEV_MODEL", "typesafe/jev-1.13")
JEV_BASE_URL = os.getenv("MEMORIA_JEV_BASE_URL", "https://openrouter.ai/api/v1")
JEV_API_KEY = os.getenv("MEMORIA_JEV_API_KEY", "")
JEV_GATEKEEPER_THRESHOLD = float(os.getenv("MEMORIA_JEV_THRESHOLD", "0.55"))

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

# Watcher & Incoming settings
DEFAULT_INCOMING_DIR = Path(os.getenv("MEMORIA_INCOMING_DIR", DEFAULT_DATA_DIR / "incoming"))
WATCH_INTERVAL = int(os.getenv("MEMORIA_WATCH_INTERVAL", "5"))
WATCH_ENABLED = os.getenv("MEMORIA_WATCH_ENABLED", "true").lower() in ("true", "1", "yes")


def ensure_data_dir() -> Path:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


def ensure_incoming_dir() -> Path:
    DEFAULT_INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_INCOMING_DIR
