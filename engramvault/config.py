import os
from pathlib import Path

def _env(name: str, default: str = "") -> str:
    return os.getenv(f"ENGRAMVAULT_{name}", os.getenv(f"MEMORIA_{name}", default))

# Versioning (Release Version + API Contract Version + MCP Protocol Version)
APP_VERSION = "2.13.1"
UI_VERSION = "2.13.1"
API_VERSION = "v1"
MCP_PROTOCOL_VERSION = "2.0"
APP_TITLE = _env("APP_TITLE", "EngramVault")
APP_SUBTITLE = _env("APP_SUBTITLE", "AI Memory Vault")

# Security & Authentication (Issue #15 & #32)
API_TOKEN = _env("API_TOKEN", "").strip()
WEB_PASSWORD = _env("WEB_PASSWORD", "").strip()
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "100"))
DEFAULT_THEME_PALETTE = _env("DEFAULT_THEME", "emerald").lower()

# Default Language & Instance White-Labeling (Issues #13 & #16)
DEFAULT_LANG = _env("DEFAULT_LANG", "en").lower()
INSTANCE_NAME = _env("INSTANCE_NAME", "Local Node")
DOCS_URL = _env("DOCS_URL", "https://engramvault.net")
DOCS_LABEL = _env("DOCS_LABEL", "Website & Docs")
REPO_URL = _env("REPO_URL", "https://github.com/gaskb/engramvault")
REPO_LABEL = _env("REPO_LABEL", "GitHub")
DESIGN_SYSTEM_URL = _env("DESIGN_SYSTEM_URL", "")

# Embedding Backend Configuration (Issue #14: 'auto', 'api', 'fastembed', 'native')
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "auto").lower()
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "")

# Optional System 1 Fast Decision Engine (Issue #18: TypeSafe Jev / OpenRouter)
JEV_ENABLED = _env("JEV_ENABLED", "false").lower() in ("true", "1", "yes")
JEV_MODEL = _env("JEV_MODEL", "typesafe/jev-1.13")
JEV_BASE_URL = _env("JEV_BASE_URL", "https://openrouter.ai/api/v1")
JEV_API_KEY = _env("JEV_API_KEY", "")
JEV_GATEKEEPER_THRESHOLD = float(_env("JEV_THRESHOLD", "0.55"))
JEV_TIMEOUT = float(_env("JEV_TIMEOUT", "12.0"))

_default_data_dir = Path.home() / ".engramvault" if not (Path.home() / ".memoria").exists() else Path.home() / ".memoria"
DEFAULT_DATA_DIR = Path(_env("DATA_DIR", str(_default_data_dir)))
DEFAULT_DB_PATH = Path(_env("DB_PATH", str(DEFAULT_DATA_DIR / "vault.db")))

# LLM Configuration
LLM_PROVIDER = _env("LLM_PROVIDER", "")
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _env("LLM_API_KEY", "dummy")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:3b")

# Chunking settings
CHUNK_SIZE = int(_env("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))

# Search weights for RRF
RRF_K = int(_env("RRF_K", "60"))
BM25_WEIGHT = float(_env("BM25_WEIGHT", "1.0"))
VEC_WEIGHT = float(_env("VEC_WEIGHT", "1.0"))

# Watcher & Incoming settings
DEFAULT_INCOMING_DIR = Path(_env("INCOMING_DIR", str(DEFAULT_DATA_DIR / "incoming")))
WATCH_INTERVAL = int(_env("WATCH_INTERVAL", "5"))
WATCH_ENABLED = _env("WATCH_ENABLED", "true").lower() in ("true", "1", "yes")


def ensure_data_dir() -> Path:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


def ensure_incoming_dir() -> Path:
    DEFAULT_INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_INCOMING_DIR
