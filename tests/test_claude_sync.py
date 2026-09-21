import json
from pathlib import Path
from unittest.mock import patch
import pytest
from memoria.db.engine import Database
from memoria.sync.claude_code import ClaudeCodeSyncer


@pytest.fixture
def sync_env(tmp_path):
    projects_dir = tmp_path / "claude_projects"
    projects_dir.mkdir()
    state_file = tmp_path / "sync_state.json"
    db_file = tmp_path / "vault.db"
    db = Database(db_file)
    return {
        "projects_dir": projects_dir,
        "state_file": state_file,
        "db": db,
    }


def test_claude_code_syncer_local(sync_env):
    projects_dir = sync_env["projects_dir"]
    state_file = sync_env["state_file"]
    db = sync_env["db"]

    # Create dummy project and session
    proj = projects_dir / "-Users-gas-project1"
    proj.mkdir()
    session1 = proj / "session1.jsonl"
    session1.write_text(
        json.dumps({"type": "user", "message": {"content": "Costruisci il frontend"}, "timestamp": "2024-07-01T10:00:00Z"}) + "\n"
        + json.dumps({"type": "assistant", "message": {"content": "Frontend configurato."}, "timestamp": "2024-07-01T10:01:00Z"}) + "\n"
    )

    syncer = ClaudeCodeSyncer(
        projects_dir=projects_dir,
        state_file=state_file,
        db=db,
    )

    # 1. Test scan
    changed = syncer.scan_changed_files()
    assert len(changed) == 1
    assert changed[0] == session1

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
    assert len(state["synced_map"] if "synced_map" in state else state["synced_files"]) == 1

    # Verify conversation is in DB
    stats = db.get_stats()
    assert stats["conversations"] == 1

    # 4. Sync again without modifications -> should report up_to_date
    res_sync2 = syncer.sync()
    assert res_sync2["status"] == "up_to_date"
    assert res_sync2["synced_count"] == 0


def test_claude_code_syncer_remote(sync_env):
    projects_dir = sync_env["projects_dir"]
    state_file = sync_env["state_file"]

    proj = projects_dir / "-Users-gas-project2"
    proj.mkdir()
    session = proj / "session2.jsonl"
    session.write_text(
        json.dumps({"type": "user", "message": {"content": "Configura il router"}, "timestamp": "2024-07-02T10:00:00Z"}) + "\n"
    )

    syncer = ClaudeCodeSyncer(
        projects_dir=projects_dir,
        state_file=state_file,
        remote_url="http://localhost:5006",
    )

    mock_resp = {
        "status": "success",
        "imported": 1,
        "skipped": 0,
        "total": 1,
    }

    with patch("memoria.sync.claude_code.post_multipart_file", return_value=mock_resp) as mock_post:
        res = syncer.sync()
        assert res["status"] == "success"
        assert res["synced_count"] == 1
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:5006/api/ingest"
