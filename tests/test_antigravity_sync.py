import json
import sqlite3
from pathlib import Path
from unittest.mock import patch
import pytest
from engramvault.db.engine import Database
from engramvault.sync.antigravity import AntigravitySyncer


@pytest.fixture
def anti_sync_env(tmp_path):
    anti_dir = tmp_path / "antigravity"
    anti_dir.mkdir()
    state_file = tmp_path / "antigravity_sync_state.json"
    db_file = tmp_path / "vault.db"
    db = Database(db_file)
    return {
        "anti_dir": anti_dir,
        "state_file": state_file,
        "db": db,
    }


def test_antigravity_syncer_local(anti_sync_env):
    anti_dir = anti_sync_env["anti_dir"]
    state_file = anti_sync_env["state_file"]
    db = anti_sync_env["db"]

    # Create dummy Antigravity brain session
    cid = "sess-12345"
    logs_dir = anti_dir / "brain" / cid / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)
    session_file = logs_dir / "transcript_full.jsonl"
    session_file.write_text(
        json.dumps({"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>Crea un workflow</USER_REQUEST>", "created_at": "2026-09-22T08:00:00Z"}) + "\n"
        + json.dumps({"step_index": 2, "type": "PLANNER_RESPONSE", "content": "Workflow completato con successo.", "created_at": "2026-09-22T08:01:00Z"}) + "\n"
    )

    syncer = AntigravitySyncer(
        antigravity_dir=anti_dir,
        state_file=state_file,
        db=db,
    )

    # 1. Test scan
    changed = syncer.scan_changed_files()
    assert len(changed) == 1
    assert changed[0] == session_file

    # 2. Test dry-run
    res_dry = syncer.sync(dry_run=True)
    assert res_dry["status"] == "dry_run"
    assert res_dry["changed_count"] == 1

    # 3. Test real local sync
    res_sync = syncer.sync()
    assert res_sync["status"] == "success"
    assert res_sync["synced_count"] == 1
    assert res_sync["error_count"] == 0

    # Verify state file saved
    assert state_file.exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(state["synced_files"]) == 1

    # Verify conversation in database
    stats = db.get_stats()
    assert stats["conversations"] == 1

    # 4. Sync again -> up_to_date
    res_sync2 = syncer.sync()
    assert res_sync2["status"] == "up_to_date"
    assert res_sync2["synced_count"] == 0


def test_antigravity_syncer_remote(anti_sync_env):
    anti_dir = anti_sync_env["anti_dir"]
    state_file = anti_sync_env["state_file"]

    cid = "sess-remote-999"
    logs_dir = anti_dir / "brain" / cid / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)
    session_file = logs_dir / "transcript_full.jsonl"
    session_file.write_text(
        json.dumps({"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>Sync verso server</USER_REQUEST>", "created_at": "2026-09-22T09:00:00Z"}) + "\n"
    )

    syncer = AntigravitySyncer(
        antigravity_dir=anti_dir,
        state_file=state_file,
        remote_url="http://localhost:5006",
    )

    mock_resp = {
        "status": "success",
        "imported": 1,
        "skipped": 0,
        "total": 1,
    }

    with patch("engramvault.sync.antigravity.post_multipart_file", return_value=mock_resp) as mock_post:
        res = syncer.sync()
        assert res["status"] == "success"
        assert res["synced_count"] == 1
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:5006/api/ingest"


def test_antigravity_syncer_exclusions(anti_sync_env, tmp_path):
    anti_dir = anti_sync_env["anti_dir"]
    state_file = anti_sync_env["state_file"]
    db = anti_sync_env["db"]

    # 1. Session in a workspace containing .no-vault sentinel file
    secret_workspace = tmp_path / "secret_project"
    secret_workspace.mkdir()
    (secret_workspace / ".no-vault").touch()

    # 2. Public workspace
    public_workspace = tmp_path / "public_project"
    public_workspace.mkdir()

    # Create dummy conversation_summaries.db
    db_path = anti_dir / "conversation_summaries.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE conversation_summaries (
            conversation_id TEXT PRIMARY KEY,
            title TEXT,
            workspace_uris TEXT
        )"""
    )
    cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
        ("sess-secret", "Secret Task", json.dumps([secret_workspace.as_uri()])),
    )
    cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
        ("sess-pattern", "Busta Paga Stipendio", json.dumps([public_workspace.as_uri()])),
    )
    cur.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
        ("sess-public", "Public App Task", json.dumps([public_workspace.as_uri()])),
    )
    conn.commit()
    conn.close()

    # Create JSONL transcripts for the three sessions
    for cid in ["sess-secret", "sess-pattern", "sess-public"]:
        ld = anti_dir / "brain" / cid / ".system_generated" / "logs"
        ld.mkdir(parents=True)
        (ld / "transcript_full.jsonl").write_text(
            json.dumps({"step_index": 1, "type": "USER_INPUT", "content": f"Prompt for {cid}"}) + "\n"
        )

    # Initialize syncer with exclude patterns (e.g. *stipendio*)
    syncer = AntigravitySyncer(
        antigravity_dir=anti_dir,
        state_file=state_file,
        db=db,
        exclude_patterns=["*stipendio*"],
    )

    # Scan should exclude sess-secret (.no-vault) and sess-pattern (matched by title/pattern)
    changed = syncer.scan_changed_files()
    assert len(changed) == 1
    assert "sess-public" in str(changed[0])

    # Perform sync
    res = syncer.sync()
    assert res["status"] == "success"
    assert res["synced_count"] == 1

    # Verify state records ignored sessions with ignored: True
    state = json.loads(state_file.read_text(encoding="utf-8"))
    synced_map = state.get("synced_files", {})
    assert any("sess-secret" in k and v.get("ignored") is True for k, v in synced_map.items())
    assert any("sess-pattern" in k and v.get("ignored") is True for k, v in synced_map.items())
    assert any("sess-public" in k and v.get("ignored") is not True for k, v in synced_map.items())

