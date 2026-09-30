import uuid
import pytest
from starlette.testclient import TestClient

from engramvault.adapters.base import NormalizedConversation, NormalizedMessage
from engramvault.db.engine import Database
from engramvault.db.models import StatementRecord
from engramvault.web.app import create_app


@pytest.fixture
def test_setup(tmp_path):
    db_file = tmp_path / "test_convs.db"
    db = Database(db_file)
    app = create_app(db_file)
    client = TestClient(app)
    return db, client


def _insert_sample_conversation(db: Database, title: str, provider: str, content: str) -> str:
    conv = NormalizedConversation(
        assistant=provider,
        title=title,
        date="2026-03-20T10:00:00Z",
        external_ref=str(uuid.uuid4()),
        messages=[
            NormalizedMessage(
                role="user",
                text=f"Hello, {content}",
                ts="2026-03-20T10:00:00Z",
            ),
            NormalizedMessage(
                role="assistant",
                text=f"This is an answer about {content}",
                ts="2026-03-20T10:01:00Z",
            ),
        ],
    )
    chunks = [
        (0, f"Hello, {content}", None),
        (1, f"This is an answer about {content}", None),
    ]
    return db.insert_conversation_with_chunks(conv, chunks)


def test_conversation_search_by_title_and_provider(test_setup):
    db, client = test_setup

    c1 = _insert_sample_conversation(db, "Guida Audiocassette Vintage", "chatgpt", "TDK MA-R Metal tape")
    c2 = _insert_sample_conversation(db, "Setup Scheda Audio Homelab", "claude", "Audient iD22 optical out")
    c3 = _insert_sample_conversation(db, "Docker Compose e Traefik", "chatgpt", "reverse proxy homelab")

    # Search by title
    res_title = db.list_conversations(query="Audiocassette")
    assert len(res_title) == 1
    assert res_title[0]["id"] == c1

    # Search by provider
    res_prov = db.list_conversations(provider="claude")
    assert len(res_prov) == 1
    assert res_prov[0]["id"] == c2

    # Combined search
    res_comb = db.list_conversations(provider="chatgpt", query="Docker")
    assert len(res_comb) == 1
    assert res_comb[0]["id"] == c3

    # Empty query returns all up to limit
    res_all = db.list_conversations()
    assert len(res_all) == 3


def test_conversation_search_by_fts_chunk_content(test_setup):
    db, client = test_setup

    c1 = _insert_sample_conversation(db, "Discussione Generale 1", "chatgpt", "chitarra elettrica Fender Stratocaster")
    c2 = _insert_sample_conversation(db, "Discussione Generale 2", "claude", "sintetizzatore analogico Korg Minilogue")

    # Search for a term that only exists in message content, not in title
    res_fender = db.list_conversations(query="Stratocaster")
    assert len(res_fender) == 1
    assert res_fender[0]["id"] == c1

    res_korg = db.list_conversations(query="Minilogue")
    assert len(res_korg) == 1
    assert res_korg[0]["id"] == c2

    # Search with no match
    res_none = db.list_conversations(query="NonExistentTermXYZ")
    assert len(res_none) == 0


def test_conversation_pagination(test_setup):
    db, client = test_setup

    for i in range(10):
        _insert_sample_conversation(db, f"Chat Numero {i:02d}", "chatgpt", f"argomento {i}")

    page1 = db.list_conversations(limit=4, offset=0)
    assert len(page1) == 4

    page2 = db.list_conversations(limit=4, offset=4)
    assert len(page2) == 4

    page3 = db.list_conversations(limit=4, offset=8)
    assert len(page3) == 2

    # Ensure page1 and page2 do not overlap
    ids_p1 = {c["id"] for c in page1}
    ids_p2 = {c["id"] for c in page2}
    assert ids_p1.isdisjoint(ids_p2)


def test_conversation_deletion_cascade(test_setup):
    db, client = test_setup

    conv_id = _insert_sample_conversation(db, "Chat da eliminare", "chatgpt", "parola segreta da cancellare")

    # Link a statement to this conversation
    ent_id = db.upsert_entity("Entità Test", "general")
    stmt = StatementRecord(
        id=str(uuid.uuid4()),
        entity_id=ent_id,
        statement_type="FACT",
        subject="Entità Test",
        predicate="owned_by_user",
        object="true",
        author="user",
        source_conversation_id=conv_id,
    )
    stmt_id = db.insert_statement(stmt)

    # Verify rows exist before deletion
    with db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations WHERE id = ?", (conv_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)).fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE conversation_id = ?", (conv_id,)).fetchone()[0] == 2

    # Delete conversation
    ok = db.delete_conversation(conv_id)
    assert ok is True

    # Verify conversation is gone
    assert db.get_conversation(conv_id) is None

    # Verify messages and chunks cascade-deleted
    with db.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks WHERE conversation_id = ?", (conv_id,)).fetchone()[0] == 0

    # Verify FTS search no longer finds the deleted conversation
    res_search = db.list_conversations(query="segreta")
    assert len(res_search) == 0

    # Verify statement still exists but source_conversation_id was SET NULL
    stmts = db.get_statements(entity_id=ent_id)
    assert len(stmts) == 1
    assert stmts[0]["id"] == stmt_id
    assert stmts[0]["source_conversation_id"] is None

    # Deleting non-existent conversation returns False
    assert db.delete_conversation("non-existent-id") is False


def test_api_conversation_search_and_delete(test_setup):
    db, client = test_setup

    c1 = _insert_sample_conversation(db, "Chat API Ricerca", "chatgpt", "contenuto unico api")

    # GET /api/conversations with search
    res = client.get("/api/conversations?q=Ricerca")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 1
    assert items[0]["id"] == c1

    # GET /api/conversations with query on message content
    res_msg = client.get("/api/conversations?q=unico")
    assert res_msg.status_code == 200
    assert len(res_msg.json()) == 1

    # DELETE /api/conversations/{id}
    res_del = client.delete(f"/api/conversations/{c1}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"
    assert res_del.json()["deleted_id"] == c1

    # Re-delete returns 404
    res_del_again = client.delete(f"/api/conversations/{c1}")
    assert res_del_again.status_code == 404


def test_vault_web_ui_delete_and_stats_sync(test_setup):
    db, client = test_setup

    c1 = _insert_sample_conversation(db, "Chat UI Test", "chatgpt", "messaggio da verificare")

    # Verify initial stats
    res_stats_before = client.get("/api/stats")
    assert res_stats_before.status_code == 200
    assert res_stats_before.json()["conversations"] == 1

    # Verify Web UI index page has required JS functions and aliases
    res_page = client.get("/")
    assert res_page.status_code == 200
    page_text = res_page.text

    assert "deleteCurrentVaultConversation" in page_text
    assert "async function fetchStats()" in page_text
    assert "window.loadStats = fetchStats" in page_text
    assert "await fetchStats();" in page_text

    # Verify that loadStats is NOT called as an unhandled or missing function
    assert "loadStats();" not in page_text

    # Simulate frontend deletion call
    res_del = client.delete(f"/api/conversations/{c1}")
    assert res_del.status_code == 200

    # Simulate fetchStats() call made immediately after deletion
    res_stats_after = client.get("/api/stats")
    assert res_stats_after.status_code == 200
    assert res_stats_after.json()["conversations"] == 0
