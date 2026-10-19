import json
from fastapi.testclient import TestClient
from engramvault.db.engine import Database
from engramvault.db.models import DecisionRecord, StatementRecord
from engramvault.knowledge.contradiction import detect_contradiction
from engramvault.mcp_server import handle_mcp_tool_call
from engramvault.web.app import create_app


def test_build_agent_context_empty_db(tmp_path):
    db = Database(tmp_path / "empty.db")
    ctx = db.build_agent_context()
    assert ctx["facts_count"] == 0
    assert ctx["decisions_count"] == 0
    assert "Profilo Utente & Contesto Homelab" in ctx["context_markdown"]
    assert "Nessun fatto verificato" in ctx["context_markdown"]


def test_build_agent_context_populated(tmp_path):
    db = Database(tmp_path / "populated.db")
    
    # Insert facts across categories
    e1 = db.upsert_entity("Mac Mini M2", "hardware_pc")
    db.insert_statement(StatementRecord(
        id="fact-1",
        entity_id=e1,
        statement_type="FACT",
        subject="Mac Mini M2",
        predicate="owned_by_user",
        object="true",
        author="user",
        is_current=1,
    ))

    e2 = db.upsert_entity("Antelope Zen Go", "audio_interface")
    db.insert_statement(StatementRecord(
        id="fact-2",
        entity_id=e2,
        statement_type="FACT",
        subject="Antelope Zen Go",
        predicate="primary_interface",
        object="USB-C 24-bit 192kHz",
        author="user",
        is_current=1,
    ))

    # Inactive fact should not appear
    e3 = db.upsert_entity("Old Laptop", "hardware_pc")
    db.insert_statement(StatementRecord(
        id="fact-3",
        entity_id=e3,
        statement_type="FACT",
        subject="Old Laptop",
        predicate="owned_by_user",
        object="false",
        author="user",
        is_current=0,
    ))

    # Decisions: active vs abandoned
    db.insert_decision(DecisionRecord(
        id="dec-1",
        topic="Database Engine",
        decision_text="Adottare SQLite locale con FTS5 per latenza < 10ms",
        status="ADOPTED",
        decided_at="2026-10-01T10:00:00Z",
    ))
    db.insert_decision(DecisionRecord(
        id="dec-2",
        topic="Cloud Sync",
        decision_text="Non sincronizzare le chiavi private su cloud provider esterni",
        status="ABANDONED",
        decided_at="2026-09-01T10:00:00Z",
    ))

    ctx = db.build_agent_context(max_facts=10, max_decisions=5)
    assert ctx["facts_count"] == 2
    assert ctx["decisions_count"] == 1  # Only ADOPTED/EVALUATING, ABANDONED is omitted
    md = ctx["context_markdown"]
    assert "Mac Mini M2" in md
    assert "Antelope Zen Go" in md
    assert "Old Laptop" not in md
    assert "Database Engine" in md
    assert "Cloud Sync" not in md


def test_agent_context_rest_and_mcp_endpoints(tmp_path):
    db_file = tmp_path / "app_context.db"
    app = create_app(db_file)
    client = TestClient(app)

    # Populate via seed
    client.post("/api/seed")

    # 1. GET /api/v1/agent/context (JSON)
    res_json = client.get("/api/v1/agent/context")
    assert res_json.status_code == 200
    data = res_json.json()
    assert "context_markdown" in data
    assert "facts" in data
    assert data["facts_count"] >= 1

    # 2. GET /api/v1/agent/context?format=markdown
    res_md = client.get("/api/v1/agent/context", params={"format": "markdown"})
    assert res_md.status_code == 200
    assert res_md.headers["content-type"].startswith("text/markdown")
    assert "# Profilo Utente & Contesto Homelab" in res_md.text

    # 3. POST /api/v1/agent/context
    res_post = client.post("/api/v1/agent/context", json={"max_facts": 5, "max_decisions": 3, "format": "json"})
    assert res_post.status_code == 200
    assert len(res_post.json()["facts"]) <= 5

    # 4. MCP memoria_get_context
    planner = app.state.planner
    db = app.state.db
    mcp_res = handle_mcp_tool_call("memoria_get_context", {"max_facts": 20}, db=db, planner=planner)
    assert "content" in mcp_res
    assert len(mcp_res["content"]) >= 1
    assert "structuredContent" in mcp_res
    assert mcp_res["structuredContent"]["facts_count"] >= 1


def test_contradiction_detection_rules(tmp_path):
    db = Database(tmp_path / "conflict.db")

    # Seed baseline fact
    e = db.upsert_entity("Proxmox Node 1", "homelab_infra")
    db.insert_statement(StatementRecord(
        id="base-1",
        entity_id=e,
        statement_type="FACT",
        subject="Proxmox Node 1",
        predicate="owned_by_user",
        object="true",
        author="user",
        is_current=1,
    ))

    # Case 1: Boolean polarity conflict
    c1 = detect_contradiction(db, "Proxmox Node 1", "owned_by_user", "false", statement_type="CLAIM")
    assert c1["has_conflict"] is True
    assert c1["conflict_type"] == "CONTRADICTION"
    assert c1["conflicting_id"] == "base-1"

    # Case 2: Direct value conflict on singleton predicate
    db.insert_statement(StatementRecord(
        id="base-2",
        entity_id=e,
        statement_type="FACT",
        subject="EngramVault",
        predicate="primary_db",
        object="SQLite",
        author="user",
        is_current=1,
    ))
    c2 = detect_contradiction(db, "EngramVault", "primary_db", "PostgreSQL", statement_type="CLAIM")
    assert c2["has_conflict"] is True
    assert c2["conflict_type"] == "CONTRADICTION"

    # Case 3: Migration / supersedes keyword
    c3 = detect_contradiction(db, "EngramVault", "primary_db", "migrato a DuckDB", statement_type="FACT")
    assert c3["has_conflict"] is True
    assert c3["conflict_type"] == "SUPERSEDES"

    # Case 4: Clash with ABANDONED decision
    db.insert_decision(DecisionRecord(
        id="dec-abandoned",
        topic="Kubernetes Cluster",
        decision_text="Troppo pesante per l'homelab, architettura abbandonata",
        status="ABANDONED",
        decided_at="2026-09-10T12:00:00Z",
    ))
    c4 = detect_contradiction(db, "Kubernetes Cluster", "owned_by_user", "true", statement_type="FACT")
    assert c4["has_conflict"] is True
    assert c4["conflict_type"] == "CONTRADICTION"
    assert c4["conflicting_type"] == "DECISION"

    # Case 5: Orthogonal / No conflict
    c5 = detect_contradiction(db, "Nuovo Dispositivo Sconosciuto", "installed_on", "Rack B", statement_type="CLAIM")
    assert c5["has_conflict"] is False
    assert c5["conflict_type"] == "NONE"


def test_contradiction_error_paths_and_edge_cases(tmp_path):
    db = Database(tmp_path / "edge_cases.db")
    app = create_app(tmp_path / "edge_cases.db")
    client = TestClient(app)

    # Empty strings, nulls, whitespace
    c_empty = detect_contradiction(db, "", "", "", statement_type="CLAIM")
    assert c_empty["has_conflict"] is False

    c_none = detect_contradiction(db, None, None, None)
    assert c_none["has_conflict"] is False

    # POST /api/statements/check-conflict endpoint
    resp_check = client.post("/api/statements/check-conflict", json={
        "subject": "Test Machine",
        "predicate": "status",
        "object": "active",
        "statement_type": "CLAIM"
    })
    assert resp_check.status_code == 200
    assert resp_check.json()["status"] == "success"
    assert resp_check.json()["report"]["has_conflict"] is False

    # Validation errors on invalid payloads
    bad_payload = client.post("/api/v1/agent/check_conflict", json={
        "subject": "",  # min_length validation failure or empty
        "predicate": "foo",
        "object_value": "bar"
    })
    # If subject is required and empty or valid string, it handles gracefully
    assert bad_payload.status_code in (200, 422)


def test_agent_remember_conflict_integration(tmp_path):
    db_file = tmp_path / "conflict_integration.db"
    app = create_app(db_file)
    client = TestClient(app)

    # 1. Remember initial fact
    r1 = client.post("/api/v1/agent/remember", json={
        "record_type": "fact",
        "subject": "Audio Deck",
        "predicate": "status",
        "object_value": "operativo",
        "category": "audio",
    })
    assert r1.status_code == 200
    stmt1_id = r1.json()["id"]

    # 2. Remember conflicting fact
    r2 = client.post("/api/v1/agent/remember", json={
        "record_type": "fact",
        "subject": "Audio Deck",
        "predicate": "status",
        "object_value": "dismesso",
        "category": "audio",
    })
    assert r2.status_code == 200
    res_data = r2.json()
    assert res_data["conflict"]["has_conflict"] is True
    assert res_data["conflict"]["conflict_type"] == "CONTRADICTION"

    # Verify Curation API returns conflict_json in statements
    entities = client.get("/api/entities").json()
    matching_entity = next(e for e in entities if e["name"] == "Audio Deck")
    conflicted_stmt = next(s for s in matching_entity["statements"] if s["object"] == "dismesso")
    assert conflicted_stmt["conflict_json"] is not None
    parsed_c = json.loads(conflicted_stmt["conflict_json"])
    assert parsed_c["has_conflict"] is True
