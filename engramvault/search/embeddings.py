import array
import hashlib
import json
import logging
import math
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

from engramvault.config import EMBEDDING_BACKEND, EMBEDDING_MODEL

logger = logging.getLogger("engramvault.search.embeddings")
EMBEDDING_DIM = 384

# Multilingual semantic concept clusters for zero-dependency cross-lingual & conceptual matching
SEMANTIC_CONCEPT_CLUSTERS: List[List[str]] = [
    ["audio", "sound", "suono", "acustico", "acoustic", "fonoassorbente", "isolamento", "studio", "mixing", "mastering"],
    ["interface", "scheda", "preamp", "preamplificatore", "converter", "convertitore", "adat", "xlr", "dsp", "fpga", "headroom", "latenza", "latency"],
    ["tape", "cassetta", "cassette", "nastro", "bias", "chrome", "cromo", "ferro", "type", "saturazione", "saturation", "analog", "analogico", "deck", "portastudio"],
    ["rock", "guitar", "chitarra", "drums", "batteria", "distortion", "distorsione", "band", "live", "brano", "track", "mix"],
    ["homelab", "server", "proxmox", "vm", "lxc", "container", "docker", "compose", "node", "nodo", "selfhosted", "cluster"],
    ["network", "rete", "dns", "pihole", "https", "ssl", "tls", "certificato", "certificate", "proxy", "nginx", "vlan", "wireguard", "vpn", "lan"],
    ["storage", "nas", "disk", "disco", "ssd", "nvme", "raid", "zfs", "backup", "smb", "nfs", "terabyte", "archivio"],
    ["ai", "llm", "agent", "agente", "mcp", "prompt", "embedding", "vector", "rag", "memory", "memoria", "knowledge", "model", "modello", "token"],
    ["inventory", "inventario", "owned", "possiedo", "posseduto", "bought", "acquistato", "sold", "venduto", "setup", "gear", "hardware", "attrezzatura"],
    ["decision", "decisione", "scelta", "choice", "adopted", "adottato", "abandoned", "scartato", "architecture", "architettura", "valutazione"],
]


def _stable_hash(text: str, seed: int = 0) -> int:
    """Deterministic cross-process hash (independent of PYTHONHASHSEED)."""
    digest = hashlib.md5(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _project_to_dim(vec: List[float], target_dim: int = EMBEDDING_DIM) -> List[float]:
    """Projects or folds any dense vector to `target_dim` and L2-normalizes it."""
    if not vec:
        return [0.0] * target_dim
    if len(vec) == target_dim:
        out = list(vec)
    else:
        out = [0.0] * target_dim
        for i, val in enumerate(vec):
            idx = i % target_dim
            sign = 1.0 if ((i // target_dim) % 2 == 0) else -1.0
            out[idx] += float(val) * sign

    norm = math.sqrt(sum(x * x for x in out))
    if norm > 0:
        out = [x / norm for x in out]
    return out


class EmbeddingEngine:
    """
    Multi-Backend Semantic Embedding Engine (Issue #14).
    Supports:
    1. 'api': Remote dense embeddings via Ollama (/api/embeddings), OpenAI-compatible (/v1/embeddings),
       or Google Gemini (embedContent with 384-dim output).
    2. 'fastembed': Local CPU ONNX dense embeddings (BAAI/bge-small or multilingual MiniLM).
    3. 'native': Deterministic Multilingual Concept-Cluster + N-gram Feature Hashing (384-dim, < 1ms).
    """

    def __init__(
        self,
        backend_mode: Optional[str] = None,
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.backend_mode = (backend or backend_mode or EMBEDDING_BACKEND or "auto").lower()
        self.model_name = model_name or EMBEDDING_MODEL or "default"
        self.api_url = (api_url or os.getenv("MEMORIA_EMBEDDING_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MEMORIA_EMBEDDING_API_KEY", "")
        self._fastembed_model = None
        self.active_backend = "native"
        self._init_backend()

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.embed_text(text_a)
        vec_b = self.embed_text(text_b)
        return float(sum(a * b for a, b in zip(vec_a, vec_b)))

    def _init_backend(self):
        if self.backend_mode in ("api", "ollama", "openai", "gemini"):
            self.active_backend = self.backend_mode
            return

        if self.backend_mode in ("auto", "fastembed"):
            try:
                from fastembed import TextEmbedding

                target_model = (
                    self.model_name
                    if self.model_name and self.model_name != "default"
                    else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                self._fastembed_model = TextEmbedding(model_name=target_model)
                self.active_backend = "fastembed"
                self.model_name = target_model
                return
            except Exception as exc:
                logger.debug("FastEmbed initialization skipped or unavailable: %s", exc)

        self.active_backend = "native"
        if not self.model_name or self.model_name == "default":
            self.model_name = "multilingual-concept-hash-384"

    def get_status(self) -> Dict[str, Any]:
        return {
            "configured_mode": self.backend_mode,
            "active_backend": self.active_backend,
            "model_name": self.model_name,
            "dimension": EMBEDDING_DIM,
            "api_url": self.api_url or None,
        }

    def _embed_via_api(self, text: str) -> Optional[List[float]]:
        """Calls external embedding endpoint (Gemini, Ollama, or OpenAI-compatible)."""
        try:
            # 1. Google Gemini Native Embedding API (S-01: pass API key via x-goog-api-key header)
            if self.active_backend == "gemini" or "generativelanguage.googleapis.com" in self.api_url:
                key = self.api_key or os.getenv("GEMINI_API_KEY", "")
                if not key:
                    return None
                model = self.model_name if self.model_name != "default" else "text-embedding-004"
                if not model.startswith("models/"):
                    model = f"models/{model}"
                url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"
                payload = {
                    "model": model,
                    "content": {"parts": [{"text": text[:4000]}]},
                    "outputDimensionality": EMBEDDING_DIM,
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": key,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    values = data.get("embedding", {}).get("values", [])
                    if values:
                        return _project_to_dim(values, EMBEDDING_DIM)

            # 2. Ollama or OpenAI-compatible /v1/embeddings
            base = self.api_url or os.getenv("MEMORIA_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
            model = self.model_name if self.model_name != "default" else "nomic-embed-text"
            endpoint = f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key != "dummy":
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = urllib.request.Request(
                endpoint,
                data=json.dumps({"model": model, "input": text[:4000]}).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                arr = data.get("data", [])
                if arr and "embedding" in arr[0]:
                    return _project_to_dim(arr[0]["embedding"], EMBEDDING_DIM)
        except Exception as exc:
            logger.debug("External embedding API call failed, falling back to local vector: %s", exc)
            return None
        return None

    def embed_text(self, text: str) -> List[float]:
        """Generates an L2-normalized 384-dimensional float vector."""
        if not text:
            return [0.0] * EMBEDDING_DIM

        # 1. Try Remote API if configured
        if self.active_backend in ("api", "ollama", "openai", "gemini"):
            api_vec = self._embed_via_api(text)
            if api_vec:
                return api_vec

        # 2. Try FastEmbed ONNX if loaded
        if self.active_backend == "fastembed" and self._fastembed_model is not None:
            try:
                vec = next(self._fastembed_model.embed([text]))
                return _project_to_dim([float(x) for x in vec], EMBEDDING_DIM)
            except Exception:
                pass

        # 3. Deterministic Multilingual Semantic Cluster + Feature Hashing Vectorizer
        vec = [0.0] * EMBEDDING_DIM
        cleaned = text.lower()
        words = re.findall(r"\b[\w\-\.]{2,}\b", cleaned)
        word_set = set(words)

        # Activate semantic concept dimensions when any synonym/concept in a cluster appears
        for cluster_idx, cluster_terms in enumerate(SEMANTIC_CONCEPT_CLUSTERS):
            matches = sum(1 for term in cluster_terms if term in word_set or term in cleaned)
            if matches > 0:
                weight = min(3.5, 1.8 + 0.5 * matches)
                for slot in range(4):
                    dim_idx = (cluster_idx * 4 + slot) % 64
                    vec[dim_idx] += weight

        # Word unigrams & bigrams (mapped across dimensions 64..383)
        span = EMBEDDING_DIM - 64
        for w in words:
            h = 64 + (_stable_hash(w, seed=1) % span)
            vec[h] += 1.2

        for i in range(len(words) - 1):
            bi = f"{words[i]}_{words[i+1]}"
            h = 64 + (_stable_hash(bi, seed=2) % span)
            vec[h] += 1.6

        # Character trigrams for morphological & cross-lingual root tolerance
        for i in range(min(len(cleaned) - 2, 1200)):
            tri = cleaned[i : i + 3]
            h = 64 + (_stable_hash(tri, seed=3) % span)
            vec[h] += 0.35

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec

    def embed_to_blob(self, text: str) -> bytes:
        vec = self.embed_text(text)
        return array.array("f", vec).tobytes()

    @staticmethod
    def blob_to_vector(blob: bytes) -> List[float]:
        if not blob:
            return []
        arr = array.array("f")
        arr.frombytes(blob)
        return arr.tolist()

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(-1.0, min(1.0, dot))


def reindex_database_embeddings(db: Any, engine: Optional[EmbeddingEngine] = None) -> Dict[str, Any]:
    """Recomputes and updates embeddings for all chunks in the database."""
    eng = engine or EmbeddingEngine()
    updated = 0
    with db.get_connection() as conn:
        rows = conn.execute("SELECT id, chunk_text FROM chunks").fetchall()
        for r in rows:
            blob = eng.embed_to_blob(r["chunk_text"] or "")
            conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (blob, r["id"]))
            updated += 1
        conn.commit()
    return {
        "status": "success",
        "reindexed_chunks": updated,
        "engine": eng.get_status(),
    }
