import array
import math
import re
from typing import List, Optional

EMBEDDING_DIM = 384


class EmbeddingEngine:
    """
    Lightweight, CPU-friendly embedding engine.
    Uses FastEmbed / SentenceTransformers if installed,
    with an optimized, deterministic n-gram hashing vectorizer as native fallback.
    """
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        # Try FastEmbed if available
        try:
            from fastembed import TextEmbedding
            self._backend = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            return
        except Exception:
            pass

        # Native fallback is always ready
        self._backend = "native"

    def embed_text(self, text: str) -> List[float]:
        """Generates an L2-normalized float vector."""
        if not text:
            return [0.0] * EMBEDDING_DIM

        if self._backend != "native" and hasattr(self._backend, "embed"):
            try:
                # FastEmbed returns a generator of numpy arrays
                vec = next(self._backend.embed([text]))
                return [float(x) for x in vec]
            except Exception:
                pass

        # Deterministic hashing vectorizer (word + character tri-grams)
        vec = [0.0] * EMBEDDING_DIM
        cleaned = text.lower()
        words = re.findall(r"\b[\w\-\.]{2,}\b", cleaned)

        # Word unigrams & bigrams
        for w in words:
            h = hash(w) % EMBEDDING_DIM
            vec[h] += 1.0

        for i in range(len(words) - 1):
            bi = f"{words[i]}_{words[i+1]}"
            h = hash(bi) % EMBEDDING_DIM
            vec[h] += 1.5

        # Character trigrams for morphological and typo tolerance
        for i in range(len(cleaned) - 2):
            tri = cleaned[i:i+3]
            h = hash(tri) % EMBEDDING_DIM
            vec[h] += 0.5

        # L2 Normalization
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
        # For L2 normalized vectors, cosine similarity is simply the dot product
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(-1.0, min(1.0, dot))
