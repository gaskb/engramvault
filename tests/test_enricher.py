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

