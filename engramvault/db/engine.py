from datetime import datetime, timezone
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("engramvault.db")
from engramvault.adapters.base import NormalizedConversation
from engramvault.db.models import (
    ConversationRecord,
    MessageRecord,
    ChunkRecord,
    EntityRecord,
    StatementRecord,
    DecisionRecord,
    RelationRecord,
)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_thread_local = threading.local()


def utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO-8601 format (YYYY-MM-DDTHH:MM:SSZ)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conns: Dict[str, sqlite3.Connection] = getattr(_thread_local, "conns", None)
        if conns is None:
            conns = {}
            _thread_local.conns = conns

        key = str(self.db_path.resolve())
        conn = conns.get(key)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                conns.pop(key, None)

        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conns[key] = conn
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            conn.executescript(schema_sql)
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
            # Backfill FTS5 tables if upgrading from pre-v2.2.0 database
            try:
                s_cnt = conn.execute("SELECT COUNT(*) FROM statements").fetchone()[0]
                sf_cnt = conn.execute("SELECT COUNT(*) FROM statements_fts").fetchone()[0]
                if s_cnt > 0 and sf_cnt == 0:
                    conn.execute(
                        "INSERT INTO statements_fts(id, subject, predicate, object, notes) "
                        "SELECT id, subject, predicate, object, COALESCE(notes, '') FROM statements"
                    )
                d_cnt = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
                df_cnt = conn.execute("SELECT COUNT(*) FROM decisions_fts").fetchone()[0]
                if d_cnt > 0 and df_cnt == 0:
                    conn.execute(
                        "INSERT INTO decisions_fts(id, topic, decision_text) "
                        "SELECT id, topic, decision_text FROM decisions"
                    )
                conn.commit()
            except Exception as exc:
                logger.debug("FTS5 backfill skipped: %s", exc)

            # Check for conflict_json migration in statements table
            try:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(statements)").fetchall()]
                if cols and "conflict_json" not in cols:
                    conn.execute("ALTER TABLE statements ADD COLUMN conflict_json TEXT DEFAULT NULL")
                    conn.commit()
            except Exception as exc:
                logger.debug("statements conflict_json migration skipped: %s", exc)

    def conversation_exists_by_hash(self, content_hash: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM conversations WHERE content_hash = ? LIMIT 1",
                (content_hash,)
            )
            return cur.fetchone() is not None

    def insert_conversation_with_chunks(
        self,
        conv: NormalizedConversation,
        chunk_texts_with_vectors: List[Tuple[int, str, Optional[bytes]]] # (msg_seq, chunk_text, vec_blob)
    ) -> str:
        """Atomically inserts a normalized conversation, its messages, and its chunks."""
        conv_id = str(uuid.uuid4())
        content_hash = conv.compute_hash()

        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, provider, external_ref, title, created_at, content_hash, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conv_id, conv.assistant, conv.external_ref, conv.title, conv.date, content_hash, json.dumps(conv.metadata, ensure_ascii=False))
            )

            msg_id_map: Dict[int, str] = {}
            for idx, msg in enumerate(conv.messages):
                msg_id = str(uuid.uuid4())
                msg_id_map[idx] = msg_id
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, ts, sequence_order, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (msg_id, conv_id, msg.role, msg.text, msg.ts, idx, json.dumps(msg.metadata, ensure_ascii=False))
                )

            for chunk_idx, (seq_order, chunk_text, vec_blob) in enumerate(chunk_texts_with_vectors):
                chunk_id = str(uuid.uuid4())
                target_msg_id = msg_id_map.get(seq_order, msg_id_map.get(0, conv_id))
                conn.execute(
                    "INSERT INTO chunks (id, message_id, conversation_id, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, target_msg_id, conv_id, chunk_idx, chunk_text, vec_blob)
                )

            conn.commit()
            return conv_id

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            c_row = cur.fetchone()
            if not c_row:
                return None

            conv_dict = dict(c_row)
            m_cur = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence_order ASC",
                (conv_id,)
            )
            conv_dict["messages"] = [dict(r) for r in m_cur.fetchall()]
            return conv_dict

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            where_clauses = []
            params: List[Any] = []

            if provider:
                where_clauses.append("c.provider = ?")
                params.append(provider)

            if query and query.strip():
                clean_q = query.strip()
                like_pattern = f"%{clean_q}%"
                fts_terms = re.findall(r"\b[\w\-\.]{2,}\b", clean_q)
                fts_query = " OR ".join([f'"{t.replace(chr(34), "").replace(chr(39), "")}"*' for t in fts_terms if t])

                if fts_query:
                    where_clauses.append(
                        """(
                            c.title LIKE ? OR c.external_ref LIKE ?
                            OR c.id IN (
                                SELECT k.conversation_id
                                FROM chunks_fts f
                                JOIN chunks k ON f.rowid = k.rowid
                                WHERE chunks_fts MATCH ?
                            )
                        )"""
                    )
                    params.extend([like_pattern, like_pattern, fts_query])
                else:
                    where_clauses.append("(c.title LIKE ? OR c.external_ref LIKE ?)")
                    params.extend([like_pattern, like_pattern])

            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            sql = f"""
                SELECT c.id, c.provider, c.external_ref, c.title, c.created_at
                FROM conversations c
                {where_sql}
                ORDER BY c.created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            try:
                cur = conn.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError as exc:
                logger.warning("FTS search fallback to LIKE in list_conversations: %s", exc)
                fb_where = []
                fb_params: List[Any] = []
                if provider:
                    fb_where.append("c.provider = ?")
                    fb_params.append(provider)
                if query and query.strip():
                    fb_where.append("(c.title LIKE ? OR c.external_ref LIKE ?)")
                    fb_params.extend([like_pattern, like_pattern])
                fb_sql = f"""
                    SELECT c.id, c.provider, c.external_ref, c.title, c.created_at
                    FROM conversations c
                    {("WHERE " + " AND ".join(fb_where)) if fb_where else ""}
                    ORDER BY c.created_at DESC
                    LIMIT ? OFFSET ?
                """
                fb_params.extend([limit, offset])
                cur = conn.execute(fb_sql, fb_params)
                return [dict(r) for r in cur.fetchall()]

    def delete_conversation(self, conv_id: str) -> bool:
        """Deletes a conversation by ID, with cascading cleanup of messages, chunks, and FTS."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()
            return cur.rowcount > 0

    # --- Knowledge Layer Operations ---

    def upsert_entity(self, name: str, category: str = "general") -> str:
        """Finds or creates an entity by normalized name.
        Sanitizes category and promotes existing 'general' category if a specific category is provided.
        """
        import re
        norm_name = name.strip().lower()
        clean_cat = re.sub(r"[^a-z0-9_]+", "_", (category or "general").strip().lower()).strip("_")
        clean_cat = clean_cat or "general"

        # Canonical category normalization (unifies synonyms and legacy misclassifications)
        canonical_map = {
            "infrastruttura": "homelab_infra",
            "homelab": "homelab_infra",
            "homelab_infra": "homelab_infra",
            "homelab_and_infra": "homelab_infra",
            "hardware": "hardware_pc",
            "hardware_pc": "hardware_pc",
            "feature": "audio_interface",
        }
        clean_cat = canonical_map.get(clean_cat, clean_cat)

        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT id, category FROM entities WHERE normalized_name = ?",
                (norm_name,)
            )
            row = cur.fetchone()
            if row:
                entity_id = row["id"]
                current_cat = (row["category"] or "").strip().lower()
                # If current category is generic/empty and new category is specific, promote it
                if current_cat in ("general", "", "none") and clean_cat not in ("general", "", "none"):
                    conn.execute("UPDATE entities SET category = ? WHERE id = ?", (clean_cat, entity_id))
                    conn.commit()
                return entity_id

            entity_id = str(uuid.uuid4())
            now_iso = utc_now_iso()
            conn.execute(
                """
                INSERT INTO entities (id, name, category, normalized_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entity_id, name.strip(), clean_cat, norm_name, now_iso)
            )
            conn.commit()
            return entity_id

    def insert_statement(self, stmt: StatementRecord) -> str:
        stmt_id = stmt.id or str(uuid.uuid4())
        sql = """INSERT INTO statements (id, entity_id, statement_type, subject, predicate, object, author, confidence, source_message_id, source_conversation_id, valid_from, valid_until, is_current, supersedes_id, notes, conflict_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        with self.get_connection() as conn:
            conn.execute(sql, (
                stmt_id, stmt.entity_id, stmt.statement_type, stmt.subject, stmt.predicate, stmt.object,
                stmt.author, stmt.confidence, stmt.source_message_id, stmt.source_conversation_id,
                stmt.valid_from, stmt.valid_until, stmt.is_current, stmt.supersedes_id, stmt.notes,
                stmt.conflict_json,
            ))
            conn.commit()
            return stmt_id

    def invalidate_statement(self, stmt_id: str, valid_until: str, supersedes_id: Optional[str] = None):
        with self.get_connection() as conn:
            conn.execute("UPDATE statements SET is_current = 0, valid_until = ? WHERE id = ?", (valid_until, stmt_id))
            conn.commit()

    def insert_decision(self, dec: DecisionRecord) -> str:
        dec_id = dec.id or str(uuid.uuid4())
        sql = """INSERT INTO decisions (id, topic, decision_text, status, decided_at, source_message_id, source_conversation_id) VALUES (?, ?, ?, ?, ?, ?, ?)"""
        with self.get_connection() as conn:
            conn.execute(sql, (
                dec_id, dec.topic, dec.decision_text, dec.status,
                dec.decided_at, dec.source_message_id, dec.source_conversation_id,
            ))
            conn.commit()
            return dec_id

    def insert_relation(self, rel: RelationRecord) -> str:
        import re
        rel_id = rel.id or str(uuid.uuid4())
        clean_type = re.sub(r"[^a-z0-9_]+", "_", (rel.relation_type or "relates_to").strip().lower()).strip("_") or "relates_to"
        sql = """INSERT OR IGNORE INTO relations (id, from_entity_id, to_entity_id, relation_type, source_message_id) VALUES (?, ?, ?, ?, ?)"""
        with self.get_connection() as conn:
            conn.execute(sql, (rel_id, rel.from_entity_id, rel.to_entity_id, clean_type, rel.source_message_id))
            conn.commit()
            # If IGNORE happened, retrieve existing id
            cur = conn.execute(
                "SELECT id FROM relations WHERE from_entity_id = ? AND to_entity_id = ? AND relation_type = ?",
                (rel.from_entity_id, rel.to_entity_id, clean_type)
            )
            row = cur.fetchone()
            return row["id"] if row else rel_id

    def delete_relation(self, rel_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM relations WHERE id = ?", (rel_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_relations(
        self,
        entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            where_clauses = ["1=1"]
            params: List[Any] = []
            if entity_id:
                where_clauses.append("(r.from_entity_id = ? OR r.to_entity_id = ?)")
                params.extend([entity_id, entity_id])
            if relation_type:
                where_clauses.append("r.relation_type = ?")
                params.append(relation_type.strip().lower())

            sql = f"""
                SELECT r.id, r.relation_type, r.source_message_id,
                       r.from_entity_id, e1.name AS from_name, e1.category AS from_category,
                       r.to_entity_id, e2.name AS to_name, e2.category AS to_category
                FROM relations r
                JOIN entities e1 ON r.from_entity_id = e1.id
                JOIN entities e2 ON r.to_entity_id = e2.id
                WHERE {" AND ".join(where_clauses)}
                ORDER BY r.rowid DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def find_related_entities(self, entity_ids: List[str], limit: int = 50) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        sql = f"""
            SELECT r.id, r.relation_type,
                   r.from_entity_id, e1.name AS from_name, e1.category AS from_category,
                   r.to_entity_id, e2.name AS to_name, e2.category AS to_category
            FROM relations r
            JOIN entities e1 ON r.from_entity_id = e1.id
            JOIN entities e2 ON r.to_entity_id = e2.id
            WHERE r.from_entity_id IN ({placeholders}) OR r.to_entity_id IN ({placeholders})
            ORDER BY r.rowid DESC
            LIMIT ?
        """
        params = list(entity_ids) + list(entity_ids) + [limit]
        with self.get_connection() as conn:
            cur = conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def get_entity_graph(self, entity_name_or_id: str, depth: int = 1) -> Dict[str, Any]:
        """Performs BFS graph traversal around an entity up to specified depth (1 or 2)."""
        target = entity_name_or_id.strip()
        norm_target = target.lower()

        with self.get_connection() as conn:
            # 1. Resolve root entity
            cur = conn.execute(
                "SELECT id, name, category, normalized_name FROM entities WHERE id = ? OR normalized_name = ?",
                (target, norm_target)
            )
            root = cur.fetchone()
            if not root:
                cur = conn.execute(
                    "SELECT id, name, category, normalized_name FROM entities WHERE name LIKE ? LIMIT 1",
                    (f"%{target}%",)
                )
                root = cur.fetchone()

            if not root:
                return {"root": None, "depth": depth, "nodes": [], "edges": [], "statements": []}

            root_dict = dict(root)
            visited_node_ids = {root_dict["id"]}
            current_frontier = {root_dict["id"]}
            collected_edges: Dict[str, Dict[str, Any]] = {}

            max_depth = max(1, min(depth, 3))
            for _ in range(max_depth):
                if not current_frontier:
                    break
                frontier_placeholders = ",".join("?" for _ in current_frontier)
                rel_sql = f"""
                    SELECT r.id, r.relation_type,
                           r.from_entity_id, e1.name AS from_name, e1.category AS from_category,
                           r.to_entity_id, e2.name AS to_name, e2.category AS to_category
                    FROM relations r
                    JOIN entities e1 ON r.from_entity_id = e1.id
                    JOIN entities e2 ON r.to_entity_id = e2.id
                    WHERE r.from_entity_id IN ({frontier_placeholders}) OR r.to_entity_id IN ({frontier_placeholders})
                """
                params = list(current_frontier) + list(current_frontier)
                cur_rels = conn.execute(rel_sql, params).fetchall()

                next_frontier = set()
                for r in cur_rels:
                    collected_edges[r["id"]] = dict(r)
                    for nid in (r["from_entity_id"], r["to_entity_id"]):
                        if nid not in visited_node_ids:
                            visited_node_ids.add(nid)
                            next_frontier.add(nid)
                current_frontier = next_frontier

            # 2. Fetch all nodes metadata
            node_placeholders = ",".join("?" for _ in visited_node_ids)
            nodes_cur = conn.execute(
                f"SELECT id, name, category, normalized_name FROM entities WHERE id IN ({node_placeholders}) ORDER BY name ASC",
                list(visited_node_ids)
            )
            nodes = [dict(r) for r in nodes_cur.fetchall()]

            # 3. Fetch current statements for these nodes
            stmts_cur = conn.execute(
                f"""
                SELECT s.id, s.entity_id, s.statement_type, s.subject, s.predicate, s.object, s.notes,
                       e.name AS entity_name, e.category AS entity_category
                FROM statements s
                JOIN entities e ON s.entity_id = e.id
                WHERE s.entity_id IN ({node_placeholders}) AND s.is_current = 1
                ORDER BY s.valid_from DESC
                LIMIT 50
                """,
                list(visited_node_ids)
            )
            stmts = [dict(s) for s in stmts_cur.fetchall()]

            return {
                "root": root_dict,
                "depth": depth,
                "nodes": nodes,
                "edges": list(collected_edges.values()),
                "statements": stmts,
            }

    def get_stats(self) -> Dict[str, int]:
        with self.get_connection() as conn:
            stats = {}
            for table in ("conversations", "messages", "chunks", "entities", "statements", "decisions", "relations"):
                cur = conn.execute(f"SELECT COUNT(*) AS c FROM {table}")
                stats[table] = cur.fetchone()["c"]

            cur = conn.execute("SELECT COUNT(*) AS c FROM statements WHERE statement_type = 'FACT'")
            stats["facts"] = cur.fetchone()["c"]

            cur = conn.execute("SELECT COUNT(*) AS c FROM statements WHERE statement_type = 'CLAIM'")
            stats["claims"] = cur.fetchone()["c"]

            return stats

    def get_entities(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = "SELECT * FROM entities"
            params = []
            if category:
                query += " WHERE category = ?"
                params.append(category)
            query += " ORDER BY name ASC"
            cur = conn.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def get_entity_categories(self) -> List[str]:
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT DISTINCT category FROM entities WHERE category IS NOT NULL AND TRIM(category) != '' ORDER BY category ASC"
            )
            return [r[0] for r in cur.fetchall()]

    def get_statements(
        self,
        entity_id: Optional[str] = None,
        statement_type: Optional[str] = None,
        is_current: Optional[int] = None,
        predicate: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = """
                SELECT s.*, e.name AS entity_name, e.category AS entity_category,
                       c.title AS conversation_title, c.provider AS conversation_provider
                FROM statements s
                LEFT JOIN entities e ON s.entity_id = e.id
                LEFT JOIN conversations c ON s.source_conversation_id = c.id
                WHERE 1=1
            """
            params = []
            if entity_id:
                query += " AND s.entity_id = ?"
                params.append(entity_id)
            if statement_type:
                query += " AND s.statement_type = ?"
                params.append(statement_type)
            if is_current is not None:
                query += " AND s.is_current = ?"
                params.append(is_current)
            if predicate:
                query += " AND s.predicate = ?"
                params.append(predicate)

            query += " ORDER BY s.valid_from DESC"
            cur = conn.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def get_decisions(
        self,
        status: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            sql = """
                SELECT d.*, c.title AS conversation_title, c.provider AS conversation_provider
                FROM decisions d
                LEFT JOIN conversations c ON d.source_conversation_id = c.id
                WHERE 1=1
            """
            params: List[Any] = []
            if status:
                sql += " AND d.status = ?"
                params.append(status.upper())
            if query and query.strip():
                clean_q = query.strip()
                sql += " AND (d.topic LIKE ? OR d.decision_text LIKE ?)"
                params.extend([f"%{clean_q}%", f"%{clean_q}%"])
            sql += " ORDER BY d.decided_at DESC"
            if limit is not None:
                sql += " LIMIT ? OFFSET ?"
                params.extend([int(limit), int(offset)])
            cur = conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_timeline(
        self,
        subject: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Returns chronological timeline of facts, claims, and decisions."""
        with self.get_connection() as conn:
            events = []
            search_term = (query or subject or "").strip()

            # Statements
            s_query = """
                SELECT s.id, 'statement' as kind, s.statement_type, s.subject, s.predicate, s.object,
                       s.valid_from as event_date, s.is_current, s.author, c.title as source_title
                FROM statements s
                LEFT JOIN conversations c ON s.source_conversation_id = c.id
            """
            s_params: List[Any] = []
            if search_term:
                s_query += " WHERE (s.subject LIKE ? OR s.predicate LIKE ? OR s.object LIKE ?)"
                pattern = f"%{search_term}%"
                s_params.extend([pattern, pattern, pattern])
            s_query += " ORDER BY s.valid_from ASC"

            for r in conn.execute(s_query, s_params).fetchall():
                events.append(dict(r))

            # Decisions
            d_query = """
                SELECT d.id, 'decision' as kind, 'DECISION' as statement_type, d.topic as subject,
                       'decided' as predicate, d.decision_text as object,
                       d.decided_at as event_date, 1 as is_current, 'user' as author, c.title as source_title
                FROM decisions d
                LEFT JOIN conversations c ON d.source_conversation_id = c.id
            """
            d_params: List[Any] = []
            if search_term:
                d_query += " WHERE (d.topic LIKE ? OR d.decision_text LIKE ?)"
                pattern = f"%{search_term}%"
                d_params.extend([pattern, pattern])
            d_query += " ORDER BY d.decided_at ASC"

            for r in conn.execute(d_query, d_params).fetchall():
                events.append(dict(r))

            events.sort(key=lambda x: x.get("event_date") or "")
            if limit is not None:
                return events[offset : offset + int(limit)]
            return events

    def update_statement(
        self,
        stmt_id: str,
        statement_type: Optional[str] = None,
        is_current: Optional[int] = None,
        notes: Optional[str] = None,
        conflict_json: Optional[str] = None,
    ) -> bool:
        """Updates statement attributes (type, status, notes, conflict_json)."""
        fields = []
        params = []
        if statement_type is not None:
            fields.append("statement_type = ?")
            params.append(statement_type)
        if is_current is not None:
            fields.append("is_current = ?")
            params.append(is_current)
        if notes is not None:
            fields.append("notes = ?")
            params.append(notes)
        if conflict_json is not None:
            fields.append("conflict_json = ?")
            params.append(conflict_json)

        if not fields:
            return False

        params.append(stmt_id)
        sql = f"UPDATE statements SET {', '.join(fields)} WHERE id = ?"
        with self.get_connection() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount > 0

    def build_agent_context(
        self,
        max_facts: int = 50,
        max_decisions: int = 20,
    ) -> Dict[str, Any]:
        """
        Builds a high-speed (<50ms), compacted user and homelab context summary
        specifically designed for LLM system prompt injection (/context command).
        """
        now_iso = utc_now_iso()
        facts = self.get_statements(statement_type="FACT", is_current=1)
        if max_facts and len(facts) > max_facts:
            facts = facts[:max_facts]

        decisions = self.get_decisions(limit=max_decisions)
        active_decisions = [
            d for d in decisions if d.get("status") in ("ADOPTED", "EVALUATING")
        ]

        # Group facts by entity category
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for f in facts:
            cat = f.get("entity_category") or "general"
            by_cat.setdefault(cat, []).append(f)

        md_lines = [
            "# Profilo Utente & Contesto Homelab (EngramVault)",
            f"- **Generato il**: {now_iso[:19].replace('T', ' ')} UTC",
            f"- **Fatti Verificati Attivi**: {len(facts)}",
            f"- **Decisioni Attive**: {len(active_decisions)}",
            "",
            "## 🖥️ Fatti Verificati & Inventario Attivo",
        ]

        if not facts:
            md_lines.append("- _Nessun fatto verificato registrato come attivo nel Vault._")
        else:
            for cat, cat_facts in sorted(by_cat.items()):
                md_lines.append(f"### Category: `{cat}`")
                for s in cat_facts:
                    pred = s.get("predicate", "owned_by_user")
                    subj = s.get("subject", "")
                    obj = s.get("object", "")
                    if pred in ("owned_by_user", "uses", "prefers") and obj in ("true", "1", "yes"):
                        md_lines.append(f"- **{subj}**")
                    else:
                        md_lines.append(f"- **{subj}** [{pred}]: {obj}")
                md_lines.append("")

        md_lines.append("## 📐 Decisioni Tecniche & Architetturali Attive")
        if not active_decisions:
            md_lines.append("- _Nessuna decisione attiva registrata nel Vault._")
        else:
            for d in active_decisions:
                st = d.get("status", "ADOPTED")
                topic = d.get("topic", "")
                text = d.get("decision_text", "")
                md_lines.append(f"- **[{st}] {topic}**: {text}")

        return {
            "generated_at": now_iso,
            "facts_count": len(facts),
            "decisions_count": len(active_decisions),
            "facts": facts,
            "decisions": active_decisions,
            "context_markdown": "\n".join(md_lines).strip(),
        }

    def delete_statement(self, stmt_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM statements WHERE id = ?", (stmt_id,))
            conn.commit()
            return cur.rowcount > 0

    def update_decision(
        self,
        dec_id: str,
        status: Optional[str] = None,
        decision_text: Optional[str] = None,
    ) -> bool:
        fields = []
        params = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if decision_text is not None:
            fields.append("decision_text = ?")
            params.append(decision_text)

        if not fields:
            return False

        params.append(dec_id)
        sql = f"UPDATE decisions SET {', '.join(fields)} WHERE id = ?"
        with self.get_connection() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount > 0

    def delete_decision(self, dec_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM decisions WHERE id = ?", (dec_id,))
            conn.commit()
            return cur.rowcount > 0

    def update_entity(
        self,
        entity_id: str,
        name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        fields = []
        params = []
        if name is not None:
            fields.append("name = ?")
            params.append(name.strip())
            fields.append("normalized_name = ?")
            params.append(name.strip().lower())
        if category is not None:
            fields.append("category = ?")
            params.append(category.strip().lower())

        if not fields:
            return False

        params.append(entity_id)
        sql = f"UPDATE entities SET {', '.join(fields)} WHERE id = ?"
        with self.get_connection() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount > 0

    def delete_entity(self, entity_id: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_unenriched_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns conversations that have not yet been processed by the batch enricher."""
        sql = """
            SELECT id, provider, title, created_at, metadata_json
            FROM conversations
            WHERE metadata_json IS NULL OR metadata_json NOT LIKE '%"enriched": true%'
            ORDER BY created_at DESC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.get_connection() as conn:
            cur = conn.execute(sql)
            return [dict(r) for r in cur.fetchall()]

    def count_unenriched_conversations(self) -> int:
        """Returns the total count of conversations that have not yet been processed by the batch enricher."""
        sql = """
            SELECT COUNT(*) AS cnt
            FROM conversations
            WHERE metadata_json IS NULL OR metadata_json NOT LIKE '%"enriched": true%'
        """
        with self.get_connection() as conn:
            cur = conn.execute(sql)
            row = cur.fetchone()
            return int(row["cnt"] if row else 0)

    def mark_conversation_enriched(self, conv_id: str, model_name: str, extracted_count: int):
        with self.get_connection() as conn:
            cur = conn.execute("SELECT metadata_json FROM conversations WHERE id = ?", (conv_id,))
            row = cur.fetchone()
            meta = {}
            if row and row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception as exc:
                    logger.debug("Invalid metadata_json for conversation %s: %s", conv_id, exc)

            meta["enriched"] = True
            meta["enriched_at"] = utc_now_iso()
            meta["extractor_model"] = model_name
            meta["extracted_count"] = extracted_count

            conn.execute(
                "UPDATE conversations SET metadata_json = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), conv_id),
            )
            conn.commit()

    def reset_knowledge_layer(self):
        """Wipes the derived Level 3 tables (statements, decisions, relations, entities) for clean rebuilds."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM statements")
            conn.execute("DELETE FROM decisions")
            conn.execute("DELETE FROM relations")
            conn.execute("DELETE FROM entities")
            # Clear enriched flags in conversations
            conn.execute(
                """
                UPDATE conversations
                SET metadata_json = json_remove(metadata_json, '$.enriched', '$.enriched_at', '$.extractor_model', '$.extracted_count')
                WHERE metadata_json LIKE '%"enriched": true%'
                """
            )
            conn.commit()

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.get_connection() as conn:
            cur = conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, now_iso),
            )
            conn.commit()

    def get_all_settings(self) -> Dict[str, str]:
        with self.get_connection() as conn:
            cur = conn.execute("SELECT key, value FROM settings")
            return {row["key"]: row["value"] for row in cur.fetchall()}

    def search_knowledge(
        self,
        query: str,
        limit: int = 15,
        category: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Searches structured knowledge using FTS5 (`statements_fts` & `decisions_fts` — R-04) with LIKE fallback."""
        import re

        stopwords = {
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da", "in",
            "con", "su", "per", "tra", "fra", "che", "chi", "cui", "non", "più", "quale",
            "quali", "quanto", "quanti", "dove", "quando", "come", "perché", "ho", "hai",
            "ha", "abbiamo", "sono", "è", "era", "stato", "del", "della", "dei", "delle",
            "nel", "nella", "nei", "nelle", "sul", "sulla", "sui", "sulle", "al", "alla",
            "the", "and", "for", "with", "about", "what", "how", "why", "from",
        }
        raw_tokens = re.findall(r"\b[\w\-\.]{2,}\b", (query or "").lower())
        tokens = [t for t in raw_tokens if t not in stopwords]
        if not tokens and raw_tokens:
            tokens = raw_tokens

        safe_fts_terms = [re.sub(r'[^a-zA-Z0-9_]', '', t) for t in tokens[:8]]
        safe_fts_terms = [t for t in safe_fts_terms if len(t) >= 2]
        fts_expr = " OR ".join(f'"{t}"*' for t in safe_fts_terms)

        with self.get_connection() as conn:
            statements: List[Dict[str, Any]] = []
            decisions: List[Dict[str, Any]] = []

            if fts_expr:
                try:
                    fts_s_sql = """
                        SELECT s.*, e.name AS entity_name, e.category AS entity_category,
                               c.title AS conversation_title, c.provider AS conversation_provider
                        FROM statements_fts f
                        JOIN statements s ON s.id = f.id
                        LEFT JOIN entities e ON s.entity_id = e.id
                        LEFT JOIN conversations c ON s.source_conversation_id = c.id
                        WHERE statements_fts MATCH ? AND s.is_current = 1
                    """
                    fts_s_params: List[Any] = [fts_expr]
                    if category:
                        fts_s_sql += " AND LOWER(e.category) = LOWER(?)"
                        fts_s_params.append(category)
                    if provider:
                        fts_s_sql += " AND LOWER(c.provider) = LOWER(?)"
                        fts_s_params.append(provider)
                    fts_s_sql += " ORDER BY CASE WHEN s.statement_type = 'FACT' THEN 0 ELSE 1 END, bm25(statements_fts) ASC LIMIT ?"
                    fts_s_params.append(limit)
                    statements = [dict(r) for r in conn.execute(fts_s_sql, fts_s_params).fetchall()]
                except Exception as exc:
                    logger.debug("statements_fts query fallback: %s", exc)

                try:
                    fts_d_sql = """
                        SELECT d.*, c.title AS conversation_title, c.provider AS conversation_provider
                        FROM decisions_fts f
                        JOIN decisions d ON d.id = f.id
                        LEFT JOIN conversations c ON d.source_conversation_id = c.id
                        WHERE decisions_fts MATCH ?
                    """
                    fts_d_params: List[Any] = [fts_expr]
                    if provider:
                        fts_d_sql += " AND LOWER(c.provider) = LOWER(?)"
                        fts_d_params.append(provider)
                    fts_d_sql += " ORDER BY bm25(decisions_fts) ASC, d.decided_at DESC LIMIT ?"
                    fts_d_params.append(limit)
                    decisions = [dict(r) for r in conn.execute(fts_d_sql, fts_d_params).fetchall()]
                except Exception as exc:
                    logger.debug("decisions_fts query fallback: %s", exc)

            if not statements:
                s_sql = """
                    SELECT s.*, e.name AS entity_name, e.category AS entity_category,
                           c.title AS conversation_title, c.provider AS conversation_provider
                    FROM statements s
                    LEFT JOIN entities e ON s.entity_id = e.id
                    LEFT JOIN conversations c ON s.source_conversation_id = c.id
                    WHERE s.is_current = 1
                """
                s_params: List[Any] = []
                if category:
                    s_sql += " AND LOWER(e.category) = LOWER(?)"
                    s_params.append(category)
                if provider:
                    s_sql += " AND LOWER(c.provider) = LOWER(?)"
                    s_params.append(provider)
                if tokens:
                    token_clauses = []
                    for t in tokens[:8]:
                        like_val = f"%{t}%"
                        token_clauses.append(
                            "(LOWER(s.subject) LIKE ? OR LOWER(s.predicate) LIKE ? OR LOWER(s.object) LIKE ? "
                            "OR LOWER(COALESCE(s.notes, '')) LIKE ? OR LOWER(COALESCE(e.category, '')) LIKE ?)"
                        )
                        s_params.extend([like_val, like_val, like_val, like_val, like_val])
                    s_sql += " AND (" + " OR ".join(token_clauses) + ")"
                s_sql += " ORDER BY CASE WHEN s.statement_type = 'FACT' THEN 0 ELSE 1 END, s.confidence DESC, s.valid_from DESC LIMIT ?"
                s_params.append(limit)
                statements = [dict(r) for r in conn.execute(s_sql, s_params).fetchall()]

            if not decisions:
                d_sql = """
                    SELECT d.*, c.title AS conversation_title, c.provider AS conversation_provider
                    FROM decisions d
                    LEFT JOIN conversations c ON d.source_conversation_id = c.id
                    WHERE 1=1
                """
                d_params: List[Any] = []
                if provider:
                    d_sql += " AND LOWER(c.provider) = LOWER(?)"
                    d_params.append(provider)
                if tokens:
                    d_clauses = []
                    for t in tokens[:8]:
                        like_val = f"%{t}%"
                        d_clauses.append(
                            "(LOWER(d.topic) LIKE ? OR LOWER(d.decision_text) LIKE ? OR LOWER(COALESCE(c.title, '')) LIKE ?)"
                        )
                        d_params.extend([like_val, like_val, like_val])
                    d_sql += " AND (" + " OR ".join(d_clauses) + ")"
                d_sql += " ORDER BY d.decided_at DESC LIMIT ?"
                d_params.append(limit)
                decisions = [dict(r) for r in conn.execute(d_sql, d_params).fetchall()]

            c_sql = "SELECT id, provider, external_ref, title, created_at FROM conversations WHERE 1=1"
            c_params: List[Any] = []
            if provider:
                c_sql += " AND LOWER(provider) = LOWER(?)"
                c_params.append(provider)
            if tokens:
                c_clauses = []
                for t in tokens[:8]:
                    c_clauses.append("LOWER(title) LIKE ?")
                    c_params.append(f"%{t}%")
                c_sql += " AND (" + " OR ".join(c_clauses) + ")"
            c_sql += " ORDER BY created_at DESC LIMIT ?"
            c_params.append(min(limit, 10))
            conversations = [dict(r) for r in conn.execute(c_sql, c_params).fetchall()]

            return {
                "tokens": tokens,
                "statements": statements,
                "decisions": decisions,
                "conversations": conversations,
            }

    def get_conversation_knowledge(self, conv_id: str) -> Dict[str, Any]:
        """Returns all statements and decisions linked to a specific conversation."""
        with self.get_connection() as conn:
            s_cur = conn.execute(
                """
                SELECT s.*, e.name AS entity_name, e.category AS entity_category
                FROM statements s
                LEFT JOIN entities e ON s.entity_id = e.id
                WHERE s.source_conversation_id = ?
                ORDER BY s.statement_type ASC, s.valid_from DESC
                """,
                (conv_id,),
            )
            d_cur = conn.execute(
                "SELECT * FROM decisions WHERE source_conversation_id = ? ORDER BY decided_at DESC",
                (conv_id,),
            )
            return {
                "statements": [dict(r) for r in s_cur.fetchall()],
                "decisions": [dict(r) for r in d_cur.fetchall()],
            }


