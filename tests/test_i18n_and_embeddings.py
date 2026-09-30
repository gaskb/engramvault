"""
Tests for v1.4.0:
- Issue #16: Extensible i18n system (English default, Italian secondary, auto-discovery)
- Issue #14: Multi-Backend Semantic Embeddings (Multilingual Concept Clusters + Re-Indexing)
- Issue #13: White-labeling & Domain-Agnostic Multi-Hop Synthesis
"""
from fastapi.testclient import TestClient

from engramvault.db.engine import Database
from engramvault.i18n import get_available_languages, get_translations, t
from engramvault.search.embeddings import EmbeddingEngine, reindex_database_embeddings
from engramvault.web.app import create_app


def test_i18n_default_english_and_locales():
    """Verify English is default and Italian translations load with full fallback."""
    langs = get_available_languages()
    codes = [item["code"] for item in langs]
    assert "en" in codes
    assert "it" in codes

    en_dict = get_translations("en")
    it_dict = get_translations("it")

    assert en_dict["ask_label"] == "Query your unified AI memory:"
    assert it_dict["ask_label"] == "Interroga la tua memoria AI unificata:"

    # Unknown language falls back cleanly to English
    fallback_dict = get_translations("fr")
    assert fallback_dict["ask_label"] == en_dict["ask_label"]

    # Helper t() works
    assert t("tab_ask", "en") == "Ask & Search"
    assert t("tab_ask", "it") == "Chiedi & Cerca"


def test_cross_lingual_semantic_embeddings():
    """Verify native semantic concept clusters align English and Italian concepts."""
    engine = EmbeddingEngine(backend="native")

    # Cross-lingual similarity between English and Italian audio/recording concepts
    sim_audio = engine.similarity(
        "recording rock music on analog cassette tape",
        "registrare brano rock su audiocassetta analogica",
    )
    sim_unrelated = engine.similarity(
        "recording rock music on analog cassette tape",
        "kubernetes docker container orchestration network",
    )
    assert sim_audio > sim_unrelated
    assert sim_audio > 0.25


def test_reindex_database_embeddings(tmp_path):
    """Verify reindex_database_embeddings updates all conversations, messages, and knowledge."""
    db_path = tmp_path / "test_reindex.db"
    db = Database(db_path=db_path)
    engine = EmbeddingEngine(backend="native")

    stats = reindex_database_embeddings(db, engine=engine)
    assert stats["status"] == "success"
    assert "reindexed_chunks" in stats
    assert stats["engine"]["active_backend"] == "native"


def test_web_i18n_and_embeddings_endpoints(tmp_path):
    """Verify /api/i18n, /api/embeddings/status, and /api/embeddings/reindex endpoints."""
    db_path = tmp_path / "test_web_v140.db"
    app = create_app(db_path=db_path)
    client = TestClient(app)

    # Default page renders in English
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "Query your unified AI memory:" in r_home.text

    # Explicit ?lang=it renders in Italian
    r_it = client.get("/?lang=it")
    assert r_it.status_code == 200
    assert "Interroga la tua memoria AI unificata:" in r_it.text

    # API i18n endpoint
    r_i18n = client.get("/api/i18n?lang=en")
    assert r_i18n.status_code == 200
    data_i18n = r_i18n.json()
    assert data_i18n["default_lang"] == "en"
    assert "locales" in data_i18n

    # Embeddings status & reindex
    r_status = client.get("/api/embeddings/status")
    assert r_status.status_code == 200
    assert "active_backend" in r_status.json()

    r_reindex = client.post("/api/embeddings/reindex")
    assert r_reindex.status_code == 200
    assert r_reindex.json()["status"] == "success"
