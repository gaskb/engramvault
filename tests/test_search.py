from pathlib import Path
import pytest

from memoria.adapters.base import NormalizedConversation, NormalizedMessage
from memoria.db.engine import Database
from memoria.search.embeddings import EmbeddingEngine
from memoria.search.hybrid import HybridSearcher, clean_fts_query


@pytest.fixture
def seeded_search_db(tmp_path: Path):
    db_file = tmp_path / "search_test.db"
    db = Database(db_file)
    embedder = EmbeddingEngine()

    c1 = NormalizedConversation(
        assistant="chatgpt",
        title="Cassette TDK e Maxell",
        date="2024-03-01T00:00:00Z",
        external_ref="1",
        messages=[
            NormalizedMessage(role="user", text="Ho comprato le TDK SA-X e le Maxell XLII."),
            NormalizedMessage(role="assistant", text="Ottimi nastri al cromo Type II per la registrazione."),
        ],
    )

    c2 = NormalizedConversation(
        assistant="claude",
        title="Registrare chitarre rock",
        date="2024-04-01T00:00:00Z",
        external_ref="2",
        messages=[
            NormalizedMessage(role="user", text="Come si registra un brano rock su nastro?"),
            NormalizedMessage(role="assistant", text="Per il rock serve saturazione calda, usa nastri ad alto bias."),
        ],
    )

    for c in [c1, c2]:
        chunks = []
        for idx, m in enumerate(c.messages):
            blob = embedder.embed_to_blob(m.text)
            chunks.append((idx, m.text, blob))
        db.insert_conversation_with_chunks(c, chunks)

    return db, embedder


def test_embedding_cosine():
    engine = EmbeddingEngine()
    v1 = engine.embed_text("registrazione brano rock chitarre")
    v2 = engine.embed_text("registrare chitarra elettrica rock nastro")
    v3 = engine.embed_text("ricetta torta alle mele e farina")

    sim1_2 = engine.cosine_similarity(v1, v2)
    sim1_3 = engine.cosine_similarity(v1, v3)

    assert sim1_2 > sim1_3
    assert sim1_2 > 0.20


def test_clean_fts_query():
    assert clean_fts_query('TDK "SA-X"!') != ""
    assert '"TDK"*' in clean_fts_query("TDK SA-X")


def test_hybrid_search(seeded_search_db):
    db, embedder = seeded_search_db
    searcher = HybridSearcher(db, embedder)

    # Search exact model name
    results = searcher.hybrid_search("TDK SA-X")
    assert len(results) > 0
    assert "TDK SA-X" in results[0]["chunk_text"]
    assert results[0]["bm25_rank"] is not None

    # Search topic
    rock_results = searcher.hybrid_search("musica rock e chitarre")
    assert len(rock_results) > 0
    assert any("rock" in r["chunk_text"].lower() for r in rock_results)
