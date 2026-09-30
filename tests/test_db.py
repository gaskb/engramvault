from pathlib import Path

from engramvault.adapters.base import NormalizedConversation, NormalizedMessage
from engramvault.db.engine import Database
from engramvault.db.models import DecisionRecord, StatementRecord


def test_db_init(tmp_path: Path):
    db = Database(tmp_path / "test_vault.db")
    stats = db.get_stats()
    assert stats["conversations"] == 0
    assert stats["messages"] == 0
    assert stats["chunks"] == 0


def test_insert_and_fts_sync(tmp_path: Path):
    db = Database(tmp_path / "test_vault.db")
    conv = NormalizedConversation(
        assistant="chatgpt",
        title="Test Titolo",
        date="2024-01-01T00:00:00Z",
        external_ref="ref-1",
        messages=[
            NormalizedMessage(role="user", text="Vorrei registrare delle chitarre"),
            NormalizedMessage(role="assistant", text="Usa un microfono a nastro."),
        ],
    )
    chunks = [
        (0, "Vorrei registrare delle chitarre", None),
        (1, "Usa un microfono a nastro.", None),
    ]

    conv_id = db.insert_conversation_with_chunks(conv, chunks)
    assert conv_id is not None

    # Verify duplicate prevention by hash
    assert db.conversation_exists_by_hash(conv.compute_hash()) is True

    # Verify FTS5 query directly
    with db.get_connection() as conn:
        res = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'chitarre'").fetchall()
        assert len(res) == 1
        res_nastro = conn.execute("SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'nastro'").fetchall()
        assert len(res_nastro) == 1


def test_statements_and_decisions_with_fts5(tmp_path: Path):
    db = Database(tmp_path / "test_vault.db")
    e_id = db.upsert_entity("Audient iD22", "audio_interface")
    assert e_id is not None

    # Add Fact
    stmt_id = db.insert_statement(
        StatementRecord(
            id=None,
            entity_id=e_id,
            statement_type="FACT",
            subject="Audient iD22",
            predicate="owned_by_user",
            object="true",
            author="user",
            valid_from="2024-01-01",
            is_current=1,
        )
    )
    assert stmt_id is not None

    # Verify FTS5 search_knowledge finds statement via statements_fts (R-04)
    k_res = db.search_knowledge("Audient", limit=5)
    assert len(k_res["statements"]) == 1
    assert k_res["statements"][0]["subject"] == "Audient iD22"

    # Invalidate
    db.invalidate_statement(stmt_id, valid_until="2026-03-01")
    curr_stmts = db.get_statements(entity_id=e_id, is_current=1)
    assert len(curr_stmts) == 0

    all_stmts = db.get_statements(entity_id=e_id)
    assert len(all_stmts) == 1
    assert all_stmts[0]["is_current"] == 0
    assert all_stmts[0]["valid_until"] == "2026-03-01"

    # Add Decision
    dec_id = db.insert_decision(
        DecisionRecord(
            id=None,
            topic="Audio Interface",
            decision_text="Passato a Zen Go",
            decided_at="2026-03-01",
        )
    )
    assert dec_id is not None
    decisions = db.get_decisions()
    assert len(decisions) == 1
    assert decisions[0]["topic"] == "Audio Interface"

    # Verify FTS5 search_knowledge finds decision via decisions_fts (R-04)
    d_res = db.search_knowledge("Zen", limit=5)
    assert len(d_res["decisions"]) == 1
    assert d_res["decisions"][0]["topic"] == "Audio Interface"
