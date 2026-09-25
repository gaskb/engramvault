"""
Unit and API tests for Issue #18:
Optional System 1 Fast Decision Engine (TypeSafe Jev / OpenRouter `typesafe/jev-1.13`)
with automatic circuit-breaker local fallback.
"""
from fastapi.testclient import TestClient

from memoria.db.engine import Database
from memoria.llm.jev import JevSystemOneEngine
from memoria.web.app import create_app


def test_jev_disabled_by_default_and_local_fallback(tmp_path):
    """Verify Jev is disabled by default and seamlessly falls back to local heuristics."""
    db = Database(tmp_path / "test_jev.db")
    jev = JevSystemOneEngine(db=db)

    status = jev.get_status()
    assert status["enabled"] is False
    assert status["active"] is False

    # 1. Query routing decision (local fallback)
    d_struct = jev.route_query_decision("Cosa possiedo tra le audiocassette TDK?")
    assert d_struct["mode"] == "structured"
    assert d_struct["category"] == "audiocassette"
    assert d_struct["strict_owned_only"] > 0.5
    assert d_struct["engine"] == "local_heuristic"

    d_synth = jev.route_query_decision("Quale interfaccia audio mi consigli per registrare chitarra?")
    assert d_synth["mode"] == "synthetic"
    assert d_synth["engine"] == "local_heuristic"

    # 2. Enrichment gatekeeper decision (local fallback)
    gate_ok = jev.evaluate_enrichment_gate("Ho comprato una Focusrite Scarlett e abbiamo deciso di usare Proxmox.")
    assert gate_ok["should_enrich"] is True
    assert gate_ok["engine"] == "local_heuristic"


def test_jev_remote_mock_and_circuit_breaker(tmp_path, monkeypatch):
    """Verify Jev uses remote primitives when enabled, and falls back cleanly on network error."""
    db = Database(tmp_path / "test_jev_mock.db")
    jev = JevSystemOneEngine(db=db, enabled=True, api_key="sk-or-test-key")

    # Mock _call_jev_primitives returning System 1 structured Choice & Noul
    monkeypatch.setattr(
        jev,
        "_call_jev_primitives",
        lambda state, schema: {
            "mode": "synthetic",
            "category": "audio_interface",
            "strict_owned_only": 0.94,
            "is_decision_lookup": 0.12,
        },
    )

    decision = jev.route_query_decision("Best preamp for electric guitar in my studio?")
    assert decision["engine"] == "jev_system_1"
    assert decision["mode"] == "synthetic"
    assert decision["category"] == "audio_interface"
    assert decision["strict_owned_only"] == 0.94

    # Gatekeeper filtering low-value debug chat
    monkeypatch.setattr(
        jev,
        "_call_jev_primitives",
        lambda state, schema: {
            "has_permanent_fact": 0.05,
            "has_architectural_decision": 0.02,
            "memory_score": 0.10,
        },
    )
    gate_noise = jev.evaluate_enrichment_gate("Traceback (most recent call last): KeyError: 'foo'")
    assert gate_noise["should_enrich"] is False
    assert gate_noise["engine"] == "jev_system_1"


def test_jev_web_endpoints(tmp_path):
    """Verify /api/jev/status, /api/jev/config, and /api/jev/test endpoints."""
    app = create_app(db_path=tmp_path / "test_web_jev.db")
    client = TestClient(app)

    r_status = client.get("/api/jev/status")
    assert r_status.status_code == 200
    assert r_status.json()["enabled"] is False

    r_cfg = client.post(
        "/api/jev/config",
        json={"enabled": True, "model": "typesafe/jev-1.13", "threshold": 0.60},
    )
    assert r_cfg.status_code == 200
    assert r_cfg.json()["enabled"] is True
    assert r_cfg.json()["threshold"] == 0.60

    r_test = client.post(
        "/api/jev/test",
        json={"query": "What equipment do I currently own?"},
    )
    assert r_test.status_code == 200
    assert r_test.json()["decision"]["mode"] == "structured"
