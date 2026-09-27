import io
import pytest
from fastapi.testclient import TestClient
from memoria.db.engine import Database
from memoria.web.app import create_app


@pytest.fixture
def client(tmp_path):
    db_file = tmp_path / "test_curation.db"
    db = Database(db_file)
    app = create_app(db_file)
    return TestClient(app)


def test_statement_curation_flow(client):
    # 1. Create manual statement
    payload = {
        "subject": "TDK MA-R",
        "predicate": "owned_by_user",
        "object": "true",
        "statement_type": "FACT",
        "entity_category": "audiocassette",
        "notes": "Acquistata su eBay",
    }
    res = client.post("/api/statements", json=payload)
    assert res.status_code == 200
    data = res.json()
    stmt_id = data["statement_id"]
    assert stmt_id is not None

    # Check in /api/entities
    res_ent = client.get("/api/entities?category=audiocassette")
    entities = res_ent.json()
    assert any(e["name"] == "TDK MA-R" for e in entities)

    # 2. Toggle statement type from FACT to CLAIM
    res_patch = client.patch(f"/api/statements/{stmt_id}", json={"statement_type": "CLAIM"})
    assert res_patch.status_code == 200

    # 3. Toggle is_current from 1 to 0 (e.g. sold)
    res_patch2 = client.patch(f"/api/statements/{stmt_id}", json={"is_current": 0})
    assert res_patch2.status_code == 200

    # Verify statement attributes updated
    res_ent2 = client.get("/api/entities?category=audiocassette")
    matching = [e for e in res_ent2.json() if e["name"] == "TDK MA-R"][0]
    stmt = matching["statements"][0]
    assert stmt["statement_type"] == "CLAIM"
    assert stmt["is_current"] == 0

    # 4. Delete statement
    res_del = client.delete(f"/api/statements/{stmt_id}")
    assert res_del.status_code == 200

    # Verify deleted
    res_ent3 = client.get("/api/entities?category=audiocassette")
    matching_after = [e for e in res_ent3.json() if e["name"] == "TDK MA-R"][0]
    assert len(matching_after["statements"]) == 0


def test_decision_curation_flow(client):
    # 1. Create manual decision
    payload = {
        "topic": "Storage Homelab",
        "decision_text": "Scelto ZFS Mirror con 2x SSD NVMe",
        "status": "ADOPTED",
        "decided_at": "2024-08-15T10:00:00Z",
    }
    res = client.post("/api/decisions", json=payload)
    assert res.status_code == 200
    dec_id = res.json()["decision_id"]

    # Check /api/decisions
    res_list = client.get("/api/decisions")
    decs = res_list.json()
    assert any(d["id"] == dec_id for d in decs)

    # 2. Update status to EVALUATING
    res_patch = client.patch(f"/api/decisions/{dec_id}", json={"status": "EVALUATING"})
    assert res_patch.status_code == 200

    # Verify update
    res_list2 = client.get("/api/decisions")
    target = [d for d in res_list2.json() if d["id"] == dec_id][0]
    assert target["status"] == "EVALUATING"

    # 3. Delete decision
    res_del = client.delete(f"/api/decisions/{dec_id}")
    assert res_del.status_code == 200

    res_list3 = client.get("/api/decisions")
    assert not any(d["id"] == dec_id for d in res_list3.json())


def test_entity_curation_flow(client):
    # Insert statement to generate an entity
    client.post("/api/statements", json={
        "subject": "FastAPI",
        "predicate": "uses",
        "object": "v0.115",
        "statement_type": "FACT",
        "entity_category": "general",
    })

    res_ent = client.get("/api/entities")
    ent = [e for e in res_ent.json() if e["name"] == "FastAPI"][0]
    ent_id = ent["id"]

    # Update category from general to software
    res_patch = client.patch(f"/api/entities/{ent_id}", json={"category": "software"})
    assert res_patch.status_code == 200

    res_ent2 = client.get("/api/entities?category=software")
    assert any(e["id"] == ent_id for e in res_ent2.json())

    # Delete entity
    res_del = client.delete(f"/api/entities/{ent_id}")
    assert res_del.status_code == 200


def test_get_categories_flow(client):
    # Insert statements with distinct categories including project and homelab
    client.post("/api/statements", json={
        "subject": "EngramVault",
        "predicate": "uses",
        "object": "true",
        "statement_type": "FACT",
        "entity_category": "project",
    })
    client.post("/api/statements", json={
        "subject": "Proxmox",
        "predicate": "runs_on",
        "object": "SER3",
        "statement_type": "FACT",
        "entity_category": "homelab",
    })

    res = client.get("/api/categories")
    assert res.status_code == 200
    cats = res.json()
    assert "project" in cats
    assert "homelab_infra" in cats
    assert "homelab" not in cats


def test_validation_and_upload_limits(client, monkeypatch):
    # 1. Statement invalid statement_type returns 422
    bad_stmt = client.post(
        "/api/statements",
        json={
            "subject": "Test",
            "predicate": "test",
            "object": "test",
            "statement_type": "INVALID_TYPE",
        },
    )
    assert bad_stmt.status_code == 422

    # 2. Decision invalid status returns 422
    bad_dec = client.post(
        "/api/decisions",
        json={
            "topic": "Test",
            "decision_text": "Text",
            "status": "INVALID_STATUS",
        },
    )
    assert bad_dec.status_code == 422

    # 3. Upload file with invalid extension returns 400
    fake_exe = io.BytesIO(b"binary content")
    res_bad_ext = client.post(
        "/api/ingest",
        files={"file": ("malware.exe", fake_exe, "application/octet-stream")},
    )
    assert res_bad_ext.status_code == 400
    assert "non supportato" in res_bad_ext.json()["detail"]

    # 4. Upload file exceeding MAX_UPLOAD_MB returns 413
    import memoria.web.routes.curation as curation_mod
    monkeypatch.setattr(curation_mod, "MAX_UPLOAD_MB", 1)  # 1 MB limit for test
    oversized = io.BytesIO(b"A" * (1024 * 1024 + 1024))  # 1 MB + 1 KB
    res_too_large = client.post(
        "/api/ingest",
        files={"file": ("big_conversations.json", oversized, "application/json")},
    )
    assert res_too_large.status_code == 413
    assert "oltre il limite" in res_too_large.json()["detail"]

