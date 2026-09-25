-- Memoria AI - SQLite Schema

PRAGMA foreign_keys = ON;

-- 1. Conversation Store
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    external_ref TEXT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content_hash TEXT UNIQUE NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_provider ON conversations(provider);
CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    ts TEXT NOT NULL,
    sequence_order INTEGER NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_role ON messages(role);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunk_conv ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunk_msg ON chunks(message_id);

-- 2. Full-Text Search (FTS5) for BM25 Lexical Matching
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers to synchronize chunks_fts automatically with chunks
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.rowid, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.rowid, old.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.rowid, old.chunk_text);
    INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.rowid, new.chunk_text);
END;

-- 3. Knowledge Layer (Entities, Facts vs Claims, Decisions, Relations)
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entity_cat ON entities(category);
CREATE INDEX IF NOT EXISTS idx_entity_norm ON entities(normalized_name);

CREATE TABLE IF NOT EXISTS statements (
    id TEXT PRIMARY KEY,
    entity_id TEXT REFERENCES entities(id) ON DELETE SET NULL,
    statement_type TEXT NOT NULL CHECK(statement_type IN ('FACT', 'CLAIM')),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    author TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    valid_from TEXT,
    valid_until TEXT,
    is_current INTEGER DEFAULT 1,
    supersedes_id TEXT REFERENCES statements(id) ON DELETE SET NULL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_stmt_entity ON statements(entity_id);
CREATE INDEX IF NOT EXISTS idx_stmt_current ON statements(is_current);
CREATE INDEX IF NOT EXISTS idx_stmt_pred_obj ON statements(predicate, object);
CREATE INDEX IF NOT EXISTS idx_stmt_subject ON statements(subject);
CREATE INDEX IF NOT EXISTS idx_stmt_subject_lower ON statements(LOWER(subject));
CREATE INDEX IF NOT EXISTS idx_stmt_object_lower ON statements(LOWER(object));

-- FTS5 Index for Knowledge Layer Statements (Issue #26 / R-04)
CREATE VIRTUAL TABLE IF NOT EXISTS statements_fts USING fts5(
    id UNINDEXED,
    subject,
    predicate,
    object,
    notes,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS statements_ai AFTER INSERT ON statements BEGIN
    INSERT INTO statements_fts(id, subject, predicate, object, notes)
    VALUES (new.id, new.subject, new.predicate, new.object, COALESCE(new.notes, ''));
END;

CREATE TRIGGER IF NOT EXISTS statements_au AFTER UPDATE ON statements BEGIN
    DELETE FROM statements_fts WHERE id = old.id;
    INSERT INTO statements_fts(id, subject, predicate, object, notes)
    VALUES (new.id, new.subject, new.predicate, new.object, COALESCE(new.notes, ''));
END;

CREATE TRIGGER IF NOT EXISTS statements_ad AFTER DELETE ON statements BEGIN
    DELETE FROM statements_fts WHERE id = old.id;
END;

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    decision_text TEXT NOT NULL,
    status TEXT DEFAULT 'ADOPTED' CHECK(status IN ('EVALUATING', 'ADOPTED', 'ABANDONED', 'SUPERSEDED')),
    decided_at TEXT NOT NULL,
    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_topic ON decisions(topic);
CREATE INDEX IF NOT EXISTS idx_decision_topic_lower ON decisions(LOWER(topic));
CREATE INDEX IF NOT EXISTS idx_decision_date ON decisions(decided_at);

-- FTS5 Index for Knowledge Layer Decisions (Issue #26 / R-04)
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    id UNINDEXED,
    topic,
    decision_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS decisions_ai AFTER INSERT ON decisions BEGIN
    INSERT INTO decisions_fts(id, topic, decision_text)
    VALUES (new.id, new.topic, new.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_au AFTER UPDATE ON decisions BEGIN
    DELETE FROM decisions_fts WHERE id = old.id;
    INSERT INTO decisions_fts(id, topic, decision_text)
    VALUES (new.id, new.topic, new.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_ad AFTER DELETE ON decisions BEGIN
    DELETE FROM decisions_fts WHERE id = old.id;
END;

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    from_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    UNIQUE(from_entity_id, to_entity_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_rel_from ON relations(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_rel_to ON relations(to_entity_id);

-- 4. Global Settings (API keys, active provider, model catalog)
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
