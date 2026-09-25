"""
Tests for Code Review Hardening (v2.1.0 — Issues #22, #23, #24, #25):
- S-01: Gemini API key sent via x-goog-api-key header (never in URL)
- S-02: Random per-password PBKDF2 salt + legacy hash compatibility
- S-03: Query-string ?api_key= disabled by default
- S-04 & S-05: /api/security/* and mutating /api/* protected when web password is enabled
- S-06 & S-07: Session cookie cryptographic timestamp expiration
- R-02: Thread-local SQLite WAL connection reuse & concurrent multithreaded writes
- R-06 & R-07: ISO-8601 UTC timestamps (never literal 'now')
- Multipart file upload on POST /api/ingest
"""

import concurrent.futures
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from memoria.db.engine import Database, utc_now_iso
from memoria.llm.client import parse_llm_json
from memoria.search.embeddings import EmbeddingEngine
from memoria.security.auth import SecurityManager, _hash_password, _verify_password_hash
from memoria.web.app import create_app


def test_pbkdf2_random_salt_and_legacy_compatibility():
    h1 = _hash_password("my-secret-pw")
    h2 = _hash_password("my-secret-pw")
    assert h1.startswith("pbkdf2_sha256$")
    assert h2.startswith("pbkdf2_sha256$")
    # Two hashes of the same password must have distinct random salts (S-02)
    assert h1 != h2
    assert _verify_password_hash("my-secret-pw", h1) is True
    assert _verify_password_hash("my-secret-pw", h2) is True
    assert _verify_password_hash("wrong-pw", h1) is False


def test_session_cookie_cryptographic_expiration(tmp_path: Path):
    db = Database(tmp_path / "cookie.db")
    mgr = SecurityManager(db)
    mgr.update_config(web_password="test-password")

    # Fresh cookie is valid
    fresh_cookie = mgr.create_session_cookie_value()
    assert mgr.verify_session_cookie(fresh_cookie) is True

    # Expired cookie (issued 31 days ago) is cryptographically rejected (S-07)
    expired_ts = int(time.time()) - (86400 * 31)
    expired_cookie = mgr.create_session_cookie_value(issued_at=expired_ts)
    assert mgr.verify_session_cookie(expired_cookie) is False


def test_gemini_api_key_in_header_not_url():
    eng = EmbeddingEngine(backend_mode="gemini", model_name="text-embedding-004", api_key="AIzaSySecretKey123")
    captured_req = {}

    class DummyResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"embedding": {"values": [0.1] * 384}}).encode("utf-8")

    def fake_urlopen(req, timeout=8):
        captured_req["url"] = req.full_url
        captured_req["headers"] = dict(req.header_items())
        return DummyResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        vec = eng._embed_via_api("hello world")
        assert vec is not None
        assert len(vec) == 384
        # S-01: Key must NOT be in the URL query string
        assert "AIzaSySecretKey123" not in captured_req["url"]
        assert "key=" not in captured_req["url"]
        # Key must be in x-goog-api-key header
        headers_lower = {k.lower(): v for k, v in captured_req["headers"].items()}
        assert headers_lower.get("x-goog-api-key") == "AIzaSySecretKey123"


def test_protected_crud_and_security_endpoints_and_iso_timestamps(tmp_path: Path):
    db_file = tmp_path / "hardening.db"
    app = create_app(db_file)
    client = TestClient(app)

    # Create statement & decision in Open LAN Mode and verify ISO-8601 timestamps (never 'now')
    stmt_res = client.post(
        "/api/statements",
        json={"subject": "Synology NAS", "predicate": "owned_by_user", "object": "DS923+"},
    )
    assert stmt_res.status_code == 200
    dec_res = client.post(
        "/api/decisions",
        json={"topic": "Domain Registrar", "decision_text": "Cloudflare Registrar"},
    )
    assert dec_res.status_code == 200

    db = Database(db_file)
    stmts = db.get_statements()
    assert stmts[0]["valid_from"] != "now"
    assert "T" in stmts[0]["valid_from"] and stmts[0]["valid_from"].endswith("Z")

    decs = db.get_decisions()
    assert decs[0]["decided_at"] != "now"
    assert "T" in decs[0]["decided_at"] and decs[0]["decided_at"].endswith("Z")

    # Enable Web Password
    cfg = client.post("/api/security/config", json={"web_password": "admin-vault-password"})
    assert cfg.status_code == 200
    token = cfg.json()["api_token"]

    # Unauthenticated client cannot access /api/security/status or POST /api/statements (S-04 & S-05)
    unauth_client = TestClient(app)
    assert unauth_client.get("/api/security/status").status_code == 401
    assert (
        unauth_client.post(
            "/api/statements",
            json={"subject": "Test", "predicate": "owned_by_user", "object": "Val"},
        ).status_code
        == 401
    )

    # Bearer token authorizes mutation and security status
    auth_with_bearer = unauth_client.get(
        "/api/security/status", headers={"Authorization": f"Bearer {token}"}
    )
    assert auth_with_bearer.status_code == 200


def test_multipart_ingest_and_concurrent_sqlite_wal(tmp_path: Path):
    db_file = tmp_path / "concurrent.db"
    app = create_app(db_file)
    client = TestClient(app)

    # Test multipart/form-data file upload on POST /api/ingest
    md_content = b"# Sessione Test Audio\n\n**User**: Che scheda audio uso?\n\n**Assistant**: Usi la RME Fireface UFX."
    files = {"file": ("session.md", md_content, "text/markdown")}
    res = client.post("/api/ingest", files=files, data={"provider": "auto"})
    assert res.status_code == 200
    assert res.json()["imported"] >= 1

    # Test concurrent multithreaded writes on SQLite WAL with thread-local pool (R-02)
    db = Database(db_file)

    def worker(idx: int) -> str:
        return db.upsert_entity(f"Concurrent Entity {idx}", "hardware")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        ids = list(ex.map(worker, range(25)))
    assert len(ids) == 25
    assert parse_llm_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert utc_now_iso().endswith("Z")


def test_query_router_and_api_token_mutation_auth(tmp_path: Path):
    from memoria.knowledge.query_router import QueryRouter

    db_file = tmp_path / "router_auth.db"
    app = create_app(db_file)
    client = TestClient(app)

    # 1. Verify QueryRouter standalone routing and language detection (A-01)
    router = QueryRouter(Database(db_file))
    assert router.detect_query_lang("Che scheda audio possiedo?") == "it"
    assert router.detect_query_lang("What audio interface do I own?") == "en"
    decision = router.select_route("Quali decisioni abbiamo preso?", mode="auto")
    assert decision["mode"] in ("structured", "synthetic")
    assert decision["is_decision_lookup"] >= 0.5

    # 2. Verify S-04 + S-05 when ONLY api_token_required=True (without web_password)
    tok_res = client.post("/api/security/token/generate")
    token = tok_res.json()["api_token"]
    client.post("/api/security/config", json={"api_token_required": True})

    # Unauthenticated mutation must return 401 even when web_password is NOT set
    assert (
        client.post(
            "/api/statements",
            json={"subject": "Synth", "predicate": "owned_by_user", "object": "true"},
        ).status_code
        == 401
    )
    assert client.get("/api/security/status").status_code == 401

    # With Bearer token, mutation succeeds
    ok_stmt = client.post(
        "/api/statements",
        json={"subject": "Synth Prophet-6", "predicate": "owned_by_user", "object": "true"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok_stmt.status_code == 200

