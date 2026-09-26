import uuid
import pytest
from starlette.testclient import TestClient

from memoria.db.engine import Database
from memoria.db.models import DecisionRecord, StatementRecord
from memoria.web.app import create_app


@pytest.fixture
def test_setup(tmp_path):
    db_file = tmp_path / "test_decisions.db"
    db = Database(db_file)
    app = create_app(db_file)
    client = TestClient(app)
    return db, client


def _insert_sample_decision(
    db: Database,
    topic: str,
    decision_text: str,
    status: str = "ADOPTED",
    decided_at: str = "2026-03-20 10:00:00",
) -> str:
    dec = DecisionRecord(
        id=str(uuid.uuid4()),
        topic=topic,
        decision_text=decision_text,
        status=status,
        decided_at=decided_at,
    )
    return db.insert_decision(dec)


def _insert_sample_statement(
    db: Database,
    subject: str,
    predicate: str,
    object_val: str,
    statement_type: str = "FACT",
    valid_from: str = "2026-03-20 10:00:00",
) -> str:
    ent_id = db.upsert_entity(subject, "general")
    stmt = StatementRecord(
        id=str(uuid.uuid4()),
        entity_id=ent_id,
        statement_type=statement_type,
        subject=subject,
        predicate=predicate,
        object=object_val,
        author="user",
        valid_from=valid_from,
    )
    return db.insert_statement(stmt)


# --- Database Layer Tests ---

def test_decision_search_by_topic_and_text(test_setup):
    db, client = test_setup

    d1 = _insert_sample_decision(db, "Audio Interface", "Adopted Audient iD22 for homelab")
    d2 = _insert_sample_decision(db, "Microphone Choice", "Selected Shure SM7B dynamic mic")
    d3 = _insert_sample_decision(db, "Tape Deck", "Serviced Nakamichi Cassette Deck 1")

    # Search by topic
    res_topic = db.get_decisions(query="Microphone")
    assert len(res_topic) == 1
    assert res_topic[0]["id"] == d2

    # Search by decision text
    res_text = db.get_decisions(query="Audient")
    assert len(res_text) == 1
    assert res_text[0]["id"] == d1

    # Search partial and case insensitive
    res_partial = db.get_decisions(query="cassette")
    assert len(res_partial) == 1
    assert res_partial[0]["id"] == d3

    # Empty search returns all
    res_all = db.get_decisions()
    assert len(res_all) == 3


def test_decision_filter_by_status(test_setup):
    db, client = test_setup

    d1 = _insert_sample_decision(db, "Arch 1", "Decision 1", status="ADOPTED")
    d2 = _insert_sample_decision(db, "Arch 2", "Decision 2", status="EVALUATING")
    d3 = _insert_sample_decision(db, "Arch 3", "Decision 3", status="ABANDONED")

    assert len(db.get_decisions(status="ADOPTED")) == 1
    assert db.get_decisions(status="ADOPTED")[0]["id"] == d1

    assert len(db.get_decisions(status="EVALUATING")) == 1
    assert db.get_decisions(status="EVALUATING")[0]["id"] == d2

    # Case insensitivity test ("abandoned" -> "ABANDONED")
    assert len(db.get_decisions(status="abandoned")) == 1
    assert db.get_decisions(status="abandoned")[0]["id"] == d3


def test_decision_combined_search_status_and_pagination(test_setup):
    db, client = test_setup

    for i in range(12):
        status = "ADOPTED" if i % 2 == 0 else "EVALUATING"
        _insert_sample_decision(db, f"Docker Task {i:02d}", f"Containerized service {i}", status=status)

    # Filter by status and search query
    adopted_tasks = db.get_decisions(status="ADOPTED", query="Docker")
    assert len(adopted_tasks) == 6

    # Test pagination
    p1 = db.get_decisions(status="ADOPTED", limit=3, offset=0)
    p2 = db.get_decisions(status="ADOPTED", limit=3, offset=3)
    assert len(p1) == 3
    assert len(p2) == 3

    p1_ids = {d["id"] for d in p1}
    p2_ids = {d["id"] for d in p2}
    assert p1_ids.isdisjoint(p2_ids)


def test_timeline_search_and_pagination(test_setup):
    db, client = test_setup

    s1 = _insert_sample_statement(db, "Mac Studio", "runs", "macOS Sequoia", valid_from="2026-01-01")
    s2 = _insert_sample_statement(db, "Synology NAS", "storage_type", "ZFS Btrfs Hybrid", valid_from="2026-01-02")
    d1 = _insert_sample_decision(db, "NAS Backup", "Use Restic backup to B2", decided_at="2026-01-03")

    # Search in timeline across statements and decisions
    res_nas = db.get_timeline(query="NAS")
    assert len(res_nas) == 2  # matches Synology NAS statement and NAS Backup decision

    res_mac = db.get_timeline(query="Mac Studio")
    assert len(res_mac) == 1
    assert res_mac[0]["id"] == s1

    # Search matching object or predicate
    res_btrfs = db.get_timeline(query="Btrfs")
    assert len(res_btrfs) == 1
    assert res_btrfs[0]["id"] == s2

    # Timeline pagination
    all_tl = db.get_timeline()
    assert len(all_tl) == 3
    p1 = db.get_timeline(limit=2, offset=0)
    p2 = db.get_timeline(limit=2, offset=2)
    assert len(p1) == 2
    assert len(p2) == 1
    assert p1[0]["id"] != p2[0]["id"]


# --- HTTP API Layer Tests ---

def test_api_decisions_endpoints(test_setup):
    db, client = test_setup

    d1 = _insert_sample_decision(db, "Database Engine", "Use SQLite WAL mode", status="ADOPTED")
    d2 = _insert_sample_decision(db, "Vector Backend", "Evaluate Qdrant or LanceDB", status="EVALUATING")

    # GET /api/decisions without query
    r = client.get("/api/decisions")
    assert r.status_code == 200
    assert len(r.json()) == 2

    # GET /api/decisions with search query
    r_search = client.get("/api/decisions?q=SQLite")
    assert r_search.status_code == 200
    items = r_search.json()
    assert len(items) == 1
    assert items[0]["id"] == d1

    # GET /api/decisions with status filter
    r_stat = client.get("/api/decisions?status=EVALUATING")
    assert r_stat.status_code == 200
    items_stat = r_stat.json()
    assert len(items_stat) == 1
    assert items_stat[0]["id"] == d2

    # PATCH /api/decisions/{id} status change
    r_patch = client.patch(f"/api/decisions/{d2}", json={"status": "ADOPTED"})
    assert r_patch.status_code == 200
    assert r_patch.json()["status"] == "success"

    # Verify updated in DB
    updated = db.get_decisions(status="ADOPTED")
    assert len(updated) == 2

    # DELETE /api/decisions/{id}
    r_del = client.delete(f"/api/decisions/{d1}")
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "success"

    # Verify deleted from DB
    remaining = db.get_decisions()
    assert len(remaining) == 1
    assert remaining[0]["id"] == d2

    # Deleting non-existent decision returns 404
    r_404 = client.delete(f"/api/decisions/{d1}")
    assert r_404.status_code == 404


def test_api_timeline_endpoints(test_setup):
    db, client = test_setup

    _insert_sample_statement(db, "Docker Homelab", "version", "27.0.1")
    _insert_sample_decision(db, "Docker Homelab Setup", "Standardize on Compose v2")

    r_all = client.get("/api/timeline")
    assert r_all.status_code == 200
    assert len(r_all.json()) == 2

    r_search = client.get("/api/timeline?q=Compose")
    assert r_search.status_code == 200
    assert len(r_search.json()) == 1


# --- Web UI Template Verification ---

def test_decisions_and_timeline_web_ui_elements(test_setup):
    db, client = test_setup

    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    # Search inputs and controls
    assert 'id="decisions-search-input"' in html
    assert 'id="decisions-search-clear"' in html
    assert 'id="decisions-status-filter"' in html
    assert 'id="decisions-load-more-container"' in html
    assert 'id="decisions-load-more-btn"' in html
    assert 'id="timeline-count-badge"' in html
    assert 'id="decisions-count-badge"' in html

    # JavaScript helper and handler functions
    assert "function debounceDecisionsSearch()" in html
    assert "function clearDecisionsSearch()" in html
    assert "async function loadDecisions(" in html
    assert "async function loadMoreDecisions()" in html
    assert "function renderDecisionsList(" in html
    assert "async function loadTimeline()" in html
    assert "async function updateDecisionStatus(" in html
    assert "async function deleteDecision(" in html
