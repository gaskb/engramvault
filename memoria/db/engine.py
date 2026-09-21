import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from memoria.adapters.base import NormalizedConversation
from memoria.db.models import (
    ConversationRecord,
    MessageRecord,
    ChunkRecord,
    EntityRecord,
    StatementRecord,
    DecisionRecord,
    RelationRecord,
)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            conn.executescript(schema_sql)
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")

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
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = "SELECT id, provider, external_ref, title, created_at FROM conversations"
            params = []
            if provider:
                query += " WHERE provider = ?"
                params.append(provider)
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cur = conn.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    # --- Knowledge Layer Operations ---

    def upsert_entity(self, name: str, category: str) -> str:
        """Finds or creates an entity by normalized name."""
        norm_name = name.strip().lower()
        with self.get_connection() as conn:
            cur = conn.execute(
                "SELECT id FROM entities WHERE normalized_name = ?",
                (norm_name,)
            )
            row = cur.fetchone()
            if row:
                return row["id"]

            entity_id = str(uuid.uuid4())
            now_iso = sqlite3.datetime.datetime.now(sqlite3.timezone.utc).isoformat() if hasattr(sqlite3, 'timezone') else "now"
            conn.execute(
                """
                INSERT INTO entities (id, name, category, normalized_name, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (entity_id, name.strip(), category.strip(), norm_name)
            )
            conn.commit()
            return entity_id

    def insert_statement(self, stmt: StatementRecord) -> str:
        stmt_id = stmt.id or str(uuid.uuid4())
        sql = """INSERT INTO statements (id, entity_id, statement_type, subject, predicate, object, author, confidence, source_message_id, source_conversation_id, valid_from, valid_until, is_current, supersedes_id, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        with self.get_connection() as conn:
            conn.execute(sql, (
                stmt_id, stmt.entity_id, stmt.statement_type, stmt.subject, stmt.predicate, stmt.object,
                stmt.author, stmt.confidence, stmt.source_message_id, stmt.source_conversation_id,
                stmt.valid_from, stmt.valid_until, stmt.is_current, stmt.supersedes_id, stmt.notes,
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

    def insert_relation(self, rel: RelationRecord):
        rel_id = rel.id or str(uuid.uuid4())
        sql = """INSERT OR IGNORE INTO relations (id, from_entity_id, to_entity_id, relation_type, source_message_id) VALUES (?, ?, ?, ?, ?)"""
        with self.get_connection() as conn:
            conn.execute(sql, (rel_id, rel.from_entity_id, rel.to_entity_id, rel.relation_type, rel.source_message_id))
            conn.commit()

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

    def get_decisions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            query = """
                SELECT d.*, c.title AS conversation_title, c.provider AS conversation_provider
                FROM decisions d
                LEFT JOIN conversations c ON d.source_conversation_id = c.id
                WHERE 1=1
            """
            params = []
            if status:
                query += " AND d.status = ?"
                params.append(status)
            query += " ORDER BY d.decided_at DESC"
            cur = conn.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def get_timeline(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns chronological timeline of facts, claims, and decisions."""
        with self.get_connection() as conn:
            events = []

            # Statements
            s_query = """
                SELECT s.id, 'statement' as kind, s.statement_type, s.subject, s.predicate, s.object,
                       s.valid_from as event_date, s.is_current, s.author, c.title as source_title
                FROM statements s
                LEFT JOIN conversations c ON s.source_conversation_id = c.id
            """
            params = []
            if subject:
                s_query += " WHERE s.subject LIKE ?"
                params.append(f"%{subject}%")
            s_query += " ORDER BY s.valid_from ASC"

            for r in conn.execute(s_query, params).fetchall():
                events.append(dict(r))

            # Decisions
            d_query = """
                SELECT d.id, 'decision' as kind, 'DECISION' as statement_type, d.topic as subject,
                       'decided' as predicate, d.decision_text as object,
                       d.decided_at as event_date, 1 as is_current, 'user' as author, c.title as source_title
                FROM decisions d
                LEFT JOIN conversations c ON d.source_conversation_id = c.id
            """
            d_params = []
            if subject:
                d_query += " WHERE d.topic LIKE ?"
                d_params.append(f"%{subject}%")
            d_query += " ORDER BY d.decided_at ASC"

            for r in conn.execute(d_query, d_params).fetchall():
                events.append(dict(r))

            events.sort(key=lambda x: x.get("event_date") or "")
            return events

    def update_statement(
        self,
        stmt_id: str,
        statement_type: Optional[str] = None,
        is_current: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Updates statement attributes (type, status, notes)."""
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

        if not fields:
            return False

        params.append(stmt_id)
        sql = f"UPDATE statements SET {', '.join(fields)} WHERE id = ?"
        with self.get_connection() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount > 0

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

    def mark_conversation_enriched(self, conv_id: str, model_name: str, extracted_count: int):
        with self.get_connection() as conn:
            cur = conn.execute("SELECT metadata_json FROM conversations WHERE id = ?", (conv_id,))
            row = cur.fetchone()
            meta = {}
            if row and row["metadata_json"]:
                try:
                    meta = json.loads(row["metadata_json"])
                except Exception:
                    pass

            meta["enriched"] = True
            meta["enriched_at"] = sqlite3.datetime.datetime.now(sqlite3.timezone.utc).isoformat() if hasattr(sqlite3, 'timezone') else "now"
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

