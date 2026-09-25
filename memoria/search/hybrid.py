import logging
import re
from typing import List, Dict, Any, Optional
from memoria.db.engine import Database
from memoria.search.embeddings import EmbeddingEngine
from memoria.config import RRF_K, BM25_WEIGHT, VEC_WEIGHT

logger = logging.getLogger("memoria.search.hybrid")

try:
    import numpy as np
    HAS_NUMPY = True
except (ImportError, RuntimeError, Exception):
    HAS_NUMPY = False


def clean_fts_query(query: str) -> str:
    """Sanitizes user input into a safe FTS5 query with prefix search."""
    tokens = re.findall(r"\b[\w\-\.]{2,}\b", query)
    if not tokens:
        return ""
    safe_terms = []
    for t in tokens:
        clean = t.replace('"', '').replace("'", "")
        if clean:
            safe_terms.append(f'"{clean}"*')
    return " OR ".join(safe_terms)


class HybridSearcher:
    def __init__(self, db: Database, embedder: Optional[EmbeddingEngine] = None):
        self.db = db
        self.embedder = embedder or EmbeddingEngine()
        self._cached_chunk_ids: Optional[List[str]] = None
        self._cached_matrix: Optional[Any] = None
        self._cached_count: int = 0

    def _ensure_vector_cache(self):
        """Loads and caches normalized vector matrix in RAM for millisecond-latency searches."""
        if not HAS_NUMPY:
            return

        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            total = cur.fetchone()[0]

            if self._cached_matrix is not None and self._cached_count == total:
                return

            cur = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
            if not rows:
                self._cached_chunk_ids = []
                self._cached_matrix = None
                self._cached_count = 0
                return

            self._cached_chunk_ids = [r[0] for r in rows]
            raw_blobs = b"".join(r[1] for r in rows)
            self._cached_matrix = np.frombuffer(raw_blobs, dtype=np.float32).reshape((len(rows), -1))
            self._cached_count = len(rows)

    def search_bm25(
        self,
        query: str,
        limit: int = 50,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        fts_query = clean_fts_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT c.id, c.chunk_text, c.message_id, c.conversation_id,
                   conv.title AS conv_title, conv.provider, conv.created_at,
                   m.role AS message_role, m.ts AS message_ts,
                   bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.rowid
            JOIN conversations conv ON c.conversation_id = conv.id
            JOIN messages m ON c.message_id = m.id
            WHERE chunks_fts MATCH ?
        """
        params = [fts_query]
        if provider:
            sql += " AND conv.provider = ?"
            params.append(provider)

        sql += " ORDER BY bm25(chunks_fts) ASC LIMIT ?"
        params.append(limit)

        with self.db.get_connection() as conn:
            try:
                cur = conn.execute(sql, params)
                results = []
                for idx, row in enumerate(cur.fetchall()):
                    d = dict(row)
                    d["bm25_rank"] = idx + 1
                    results.append(d)
                return results
            except Exception as exc:
                logger.debug("FTS5 search warning for query '%s': %s", fts_query, exc)
                return []

    def search_vector(
        self,
        query: str,
        limit: int = 50,
        candidate_ids: Optional[set[str]] = None,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        q_vec = self.embedder.embed_text(query)

        if HAS_NUMPY:
            self._ensure_vector_cache()
            if self._cached_matrix is not None and len(self._cached_chunk_ids) > 0:
                q_arr = np.array(q_vec, dtype=np.float32)
                scores = self._cached_matrix.dot(q_arr)

                # Filter by candidate_ids or top scores
                top_k = min(limit * 2, len(scores))
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                sorted_indices = top_indices[np.argsort(-scores[top_indices])]

                selected_ids = []
                scores_map = {}
                for idx in sorted_indices:
                    cid = self._cached_chunk_ids[idx]
                    if candidate_ids and cid not in candidate_ids:
                        continue
                    sim = float(scores[idx])
                    if sim > 0.05:
                        selected_ids.append(cid)
                        scores_map[cid] = sim
                    if len(selected_ids) >= limit:
                        break

                if not selected_ids:
                    return []

                # Fetch details for selected top chunks
                placeholders = ",".join("?" for _ in selected_ids)
                sql = f"""
                    SELECT c.id, c.chunk_text, c.message_id, c.conversation_id,
                           conv.title AS conv_title, conv.provider, conv.created_at,
                           m.role AS message_role, m.ts AS message_ts
                    FROM chunks c
                    JOIN conversations conv ON c.conversation_id = conv.id
                    JOIN messages m ON c.message_id = m.id
                    WHERE c.id IN ({placeholders})
                """
                params = list(selected_ids)
                if provider:
                    sql += " AND conv.provider = ?"
                    params.append(provider)

                with self.db.get_connection() as conn:
                    cur = conn.execute(sql, params)
                    results = []
                    for row in cur:
                        d = dict(row)
                        d["vector_similarity"] = scores_map.get(d["id"], 0.0)
                        results.append(d)
                    results.sort(key=lambda x: x["vector_similarity"], reverse=True)
                    for r_idx, item in enumerate(results):
                        item["vector_rank"] = r_idx + 1
                    return results

        # Pure python fallback
        sql = """
            SELECT c.id, c.chunk_text, c.message_id, c.conversation_id, c.embedding,
                   conv.title AS conv_title, conv.provider, conv.created_at,
                   m.role AS message_role, m.ts AS message_ts
            FROM chunks c
            JOIN conversations conv ON c.conversation_id = conv.id
            JOIN messages m ON c.message_id = m.id
        """
        params = []
        if provider:
            sql += " WHERE conv.provider = ?"
            params.append(provider)

        scored_items = []
        with self.db.get_connection() as conn:
            cur = conn.execute(sql, params)
            for row in cur:
                cid = row["id"]
                if candidate_ids and cid not in candidate_ids:
                    continue

                blob = row["embedding"]
                if not blob:
                    continue

                doc_vec = self.embedder.blob_to_vector(blob)
                sim = self.embedder.cosine_similarity(q_vec, doc_vec)
                if sim > 0.05:
                    item = dict(row)
                    del item["embedding"]
                    item["vector_similarity"] = sim
                    scored_items.append(item)

        scored_items.sort(key=lambda x: x["vector_similarity"], reverse=True)
        results = []
        for idx, item in enumerate(scored_items[:limit]):
            item["vector_rank"] = idx + 1
            results.append(item)
        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 15,
        provider: Optional[str] = None,
        rrf_k: int = RRF_K,
        w_bm25: float = BM25_WEIGHT,
        w_vec: float = VEC_WEIGHT
    ) -> List[Dict[str, Any]]:
        """Executes Hybrid Search with Reciprocal Rank Fusion (RRF)."""
        # 1. Lexical BM25 Search
        bm25_hits = self.search_bm25(query, limit=50, provider=provider)
        bm25_map = {hit["id"]: hit for hit in bm25_hits}

        # 2. Semantic Vector Search
        vector_hits = self.search_vector(query, limit=50, provider=provider)
        vector_map = {hit["id"]: hit for hit in vector_hits}

        # 3. Fuse via RRF
        all_ids = set(bm25_map.keys()).union(set(vector_map.keys()))
        fused_scores = {}

        for cid in all_ids:
            score = 0.0
            b_hit = bm25_map.get(cid)
            v_hit = vector_map.get(cid)

            if b_hit:
                score += w_bm25 / (rrf_k + b_hit["bm25_rank"])
            if v_hit:
                score += w_vec / (rrf_k + v_hit["vector_rank"])

            base_item = b_hit or v_hit
            item_copy = dict(base_item)
            item_copy["rrf_score"] = round(score, 6)
            item_copy["bm25_rank"] = b_hit["bm25_rank"] if b_hit else None
            item_copy["vector_rank"] = v_hit["vector_rank"] if v_hit else None
            item_copy["vector_similarity"] = v_hit["vector_similarity"] if v_hit else None
            fused_scores[cid] = item_copy

        sorted_results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:limit]
