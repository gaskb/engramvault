import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from engramvault.db.engine import Database
from engramvault.knowledge.enricher import KnowledgeEnricher
from engramvault.llm.base import LLMResponse
from engramvault.llm.client import LLMClient
from engramvault.seed.seed_data import seed_database
from engramvault.web.app import create_app
from starlette.testclient import TestClient


@pytest.fixture
def populated_db(tmp_path):
    db_file = tmp_path / "test_enrich.db"
    db = Database(db_file)
    seed_database(db)
    return db


def test_knowledge_enricher_heuristics(populated_db):
    enricher = KnowledgeEnricher(populated_db, llm=None, delay_between_calls=0)

    # 1. Run batch without LLM (uses robust heuristic extractor fallback)
    summary = enricher.run_batch(limit=10)
    assert summary["processed"] > 0
    assert summary["error"] is None

    # Check unenriched count decreased
    remaining = populated_db.get_unenriched_conversations()
    assert len(remaining) == 0

    # 2. Run batch again -> should process 0 conversations (resume capability)
    summary2 = enricher.run_batch()
    assert summary2["total_to_process"] == 0
    assert summary2["processed"] == 0


def test_knowledge_enricher_with_llm(populated_db):
    mock_client = MagicMock(spec=LLMClient)
    mock_client.is_configured.return_value = True
    mock_client.model = "mock-flash-extractor"

    llm_payload = {
        "entities": [{"name": "Audient iD22", "category": "audio_interface"}],
        "facts": [{"subject": "Audient iD22", "predicate": "owned_by_user", "object": "true"}],
        "claims": [{"subject": "Audient iD22", "predicate": "hardware_quality", "object": "Preamplificatori discreti classe A"}],
        "decisions": [{"topic": "Interfaccia", "decision": "Usare come scheda principale", "status": "ADOPTED"}],
    }

    mock_client.chat_completion.return_value = f"```json\n{json.dumps(llm_payload)}\n```"

    enricher = KnowledgeEnricher(populated_db, llm=mock_client, delay_between_calls=0)

    # Run batch with rebuild=True
    summary = enricher.run_batch(limit=2, rebuild=True)
    assert summary["processed"] == 2
    assert summary["new_facts"] >= 2
    assert summary["new_decisions"] >= 2

    # Verify statements are recorded
    stmts = populated_db.get_statements(predicate="hardware_quality")
    assert len(stmts) >= 1
    assert stmts[0]["statement_type"] == "CLAIM"


def test_enrich_web_api(populated_db, tmp_path):
    app = create_app(populated_db.db_path)
    client = TestClient(app)

    # Check initial status
    res_status = client.get("/api/enrich/status")
    assert res_status.status_code == 200
    assert res_status.json()["is_running"] is False

    # Trigger start enrich
    res_start = client.post("/api/enrich/start", json={"limit": 3, "delay": 0.0})
    assert res_start.status_code == 200
    assert res_start.json()["status"] in ("started", "already_running")

    # Stop enrich
    res_stop = client.post("/api/enrich/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "stopped"


def test_category_sanitization_and_promotion_and_open_world(populated_db):
    # 1. Sanitization and promotion of general category
    ent_id1 = populated_db.upsert_entity("Yaesu FT-891", "general")
    entities = populated_db.get_entities()
    target = next(e for e in entities if e["id"] == ent_id1)
    assert target["category"] == "general"

    # Promote to specific category with weird casing/characters
    ent_id2 = populated_db.upsert_entity("Yaesu FT-891", "Ham Radio / Transceiver!")
    assert ent_id1 == ent_id2
    entities = populated_db.get_entities()
    target = next(e for e in entities if e["id"] == ent_id1)
    assert target["category"] == "ham_radio_transceiver"

    # Should not downgrade back to general
    populated_db.upsert_entity("Yaesu FT-891", "general")
    entities = populated_db.get_entities()
    target = next(e for e in entities if e["id"] == ent_id1)
    assert target["category"] == "ham_radio_transceiver"

    # 2. Enricher LLM propagation of dynamic categories to statements
    mock_client = MagicMock(spec=LLMClient)
    mock_client.is_configured.return_value = True
    mock_client.model = "mock-dynamic-extractor"

    llm_payload = {
        "entities": [{"name": "Kenwood TS-590SG", "category": "ham_radio"}],
        "facts": [{"subject": "Kenwood TS-590SG", "predicate": "owned_by_user", "object": "true"}],
        "claims": [],
        "decisions": [],
    }
    mock_client.chat_completion.return_value = json.dumps(llm_payload)

    enricher = KnowledgeEnricher(populated_db, llm=mock_client, delay_between_calls=0)
    summary = enricher.run_batch(limit=1, rebuild=True)
    assert summary["processed"] == 1

    stmts = populated_db.get_statements(predicate="owned_by_user")
    kenwood_stmt = next(s for s in stmts if s["subject"] == "Kenwood TS-590SG")
    assert kenwood_stmt["entity_category"] == "ham_radio"


def test_parse_llm_json_malformed():
    from engramvault.llm.client import parse_llm_json
    # Malformed JSON should gracefully return None without raising NameError or JSONDecodeError
    assert parse_llm_json("{invalid json") is None
    assert parse_llm_json(None) is None
    assert parse_llm_json("```json\n{\"test\": 1\n") is None


def test_enricher_recovers_from_single_conversation_failure(populated_db):
    mock_client = MagicMock(spec=LLMClient)
    mock_client.is_configured.return_value = True
    mock_client.model = "mock-resilient"

    # First call returns invalid JSON, second returns valid JSON
    mock_client.chat_completion.side_effect = [
        "Unterminated json {...",
        json.dumps({"entities": [{"name": "Device B", "category": "hardware"}], "facts": []}),
    ]

    enricher = KnowledgeEnricher(populated_db, llm=mock_client, delay_between_calls=0)
    summary = enricher.run_batch(limit=2, rebuild=True)
    assert summary["processed"] == 2
    assert summary["error"] is None


def test_enricher_handles_llm_timeout(populated_db):
    """Verifies that an external LLM timeout (or connection drop) is caught without crashing the batch."""
    import socket
    mock_client = MagicMock(spec=LLMClient)
    mock_client.is_configured.return_value = True
    mock_client.model = "mock-timeout"

    # Simulate timeout on first call, success on second
    mock_client.chat_completion.side_effect = [
        TimeoutError("Connection to LLM provider timed out after 30s"),
        json.dumps({
            "entities": [{"name": "Server Backup", "category": "homelab_infra"}],
            "facts": [{"subject": "Server Backup", "predicate": "uses", "object": "ZFS"}],
            "claims": [],
            "decisions": [],
        }),
    ]

    enricher = KnowledgeEnricher(populated_db, llm=mock_client, delay_between_calls=0)
    summary = enricher.run_batch(limit=2, rebuild=True)

    assert summary["processed"] == 2
    assert summary["error"] is None
    assert summary["new_facts"] >= 1


def test_enricher_corrupted_unicode_and_null_bytes(populated_db):
    """Verifies that LLM responses with corrupted Unicode surrogates or null bytes are handled cleanly."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.is_configured.return_value = True
    mock_client.model = "mock-unicode"

    # LLM returns payload with null bytes and invalid entity structure
    mock_client.chat_completion.return_value = json.dumps({
        "entities": [
            {"name": "Nakamichi \x00 Cassette Deck \ud83d", "category": "hardware_deck"},
            {"name": "", "category": None},
        ],
        "facts": [
            {"subject": "Nakamichi \x00 Cassette Deck", "predicate": "owned_by_user", "object": "true"},
            {"subject": "", "predicate": "invalid", "object": None},
        ],
        "claims": [
            {"subject": "Audio Quality", "predicate": None, "object": "Eccellente \x00"},
        ],
        "decisions": [
            {"topic": "Manutenzione", "decision": "Sostituire pinch roller", "status": "UNKNOWN_STATUS"},
            {"topic": "", "decision": ""},
        ],
        "relations": [
            {"from_entity": "Nakamichi", "relation": "connected_to", "to_entity": "Audient"},
            {"from_entity": "Same", "to_entity": "Same"},  # self-loop should be ignored
        ],
    })

    enricher = KnowledgeEnricher(populated_db, llm=mock_client, delay_between_calls=0)
    summary = enricher.run_batch(limit=1, rebuild=True)

    assert summary["processed"] == 1
    assert summary["error"] is None
    assert summary["new_facts"] >= 1
    assert summary["new_decisions"] >= 1
    # Unknown status should be safely normalized to ADOPTED
    decs = populated_db.get_decisions()
    assert any(d["status"] == "ADOPTED" and "pinch roller" in d["decision_text"] for d in decs)


def test_enricher_environment_decoupling(monkeypatch, tmp_path):
    """Ensures test suite runs successfully without depending on developer local .env secrets."""
    # Wipe all relevant environment variables
    for key in ["ENGRAMVAULT_API_TOKEN", "ENGRAMVAULT_WEB_PASSWORD", "MEMORIA_API_TOKEN",
                "MEMORIA_WEB_PASSWORD", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    db_path = tmp_path / "decoupled_vault.db"
    db = Database(db_path)
    seed_database(db)

    # When LLM client is not configured (or explicitly mock unconfigured), enricher must fall back to heuristics
    mock_unconfigured = MagicMock(spec=LLMClient)
    mock_unconfigured.is_configured.return_value = False
    mock_unconfigured.model = "none"

    enricher = KnowledgeEnricher(db, llm=mock_unconfigured, delay_between_calls=0)
    assert enricher.llm.is_configured() is False

    summary = enricher.run_batch(rebuild=True)
    assert summary["processed"] >= 5
    assert summary["error"] is None
    # All processed conversations should be marked enriched in DB
    remaining = db.get_unenriched_conversations()
    assert len(remaining) == 0


