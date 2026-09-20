import re
from typing import List, Dict, Any, Optional
from memoria.db.engine import Database
from memoria.search.embeddings import EmbeddingEngine
from memoria.config import RRF_K, BM25_WEIGHT, VEC_WEIGHT


def clean_fts_query(query: str) -> str:
    """Sanitizes user input into a safe FTS5 query with prefix search."""
    tokens = re.findall(r"\b[\w\-\.]{2,}\b", query)
    if not tokens:
        return ""
    # Use OR or AND; AND with prefix matching works best for finding relevant chunks
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
            except Exception:
                return []

    def search_vector(
        self,
        query: str,
        limit: int = 50,
        candidate_ids: Optional[set[str]] = None,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        q_vec = self.embedder.embed_text(query)

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
                if sim > 0.05:  # threshold
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

            # Base item info from whichever hit found it
            base_item = b_hit or v_hit
            item_copy = dict(base_item)
            item_copy["rrf_score"] = round(score, 6)
            item_copy["bm25_rank"] = b_hit["bm25_rank"] if b_hit else None
            item_copy["vector_rank"] = v_hit["vector_rank"] if v_hit else None
            item_copy["vector_similarity"] = v_hit["vector_similarity"] if v_hit else None
            fused_scores[cid] = item_copy

        sorted_results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:limit]
