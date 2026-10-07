import json
from pathlib import Path
from unittest.mock import patch
import pytest
from engramvault.db.engine import Database
from engramvault.sync.claude_code import ClaudeCodeSyncer


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

    with patch("engramvault.sync.claude_code.post_multipart_file", return_value=mock_resp) as mock_post:
        res = syncer.sync()
        assert res["status"] == "success"
        assert res["synced_count"] == 1
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:5006/api/ingest"


def test_claude_code_syncer_exclusions(sync_env, tmp_path):
    projects_dir = sync_env["projects_dir"]
    state_file = sync_env["state_file"]
    db = sync_env["db"]

    # 1. Project folder matching an exclude pattern
    proj_sensitive = projects_dir / "-Users-gas-dev-banking-app"
    proj_sensitive.mkdir()
    sess_sensitive = proj_sensitive / "session_sens.jsonl"
    sess_sensitive.write_text(
        json.dumps({"type": "user", "message": {"content": "Conto corrente"}, "timestamp": "2024-07-03T10:00:00Z"}) + "\n"
    )

    # 2. Project containing cwd with .no-vault sentinel file on disk
    real_workdir = tmp_path / "finance_workdir"
    real_workdir.mkdir()
    (real_workdir / ".no-vault").touch()

    proj_sentinel = projects_dir / "-Users-gas-dev-finance"
    proj_sentinel.mkdir()
    sess_sentinel = proj_sentinel / "session_sent.jsonl"
    sess_sentinel.write_text(
        json.dumps({
            "type": "user",
            "cwd": str(real_workdir),
            "message": {"content": "Bilancio annuale"},
            "timestamp": "2024-07-03T11:00:00Z",
        }) + "\n"
    )

    # 3. Standard public project
    proj_public = projects_dir / "-Users-gas-dev-public-docs"
    proj_public.mkdir()
    sess_public = proj_public / "session_pub.jsonl"
    sess_public.write_text(
        json.dumps({"type": "user", "message": {"content": "Scrivi documentazione"}, "timestamp": "2024-07-03T12:00:00Z"}) + "\n"
    )

    syncer = ClaudeCodeSyncer(
        projects_dir=projects_dir,
        state_file=state_file,
        db=db,
        exclude_patterns=["*banking*"],
    )

    # Scan should exclude banking (pattern) and finance (sentinel in cwd)
    changed = syncer.scan_changed_files()
    assert len(changed) == 1
    assert changed[0] == sess_public

    # Sync
    res = syncer.sync()
    assert res["status"] == "success"
    assert res["synced_count"] == 1

    # Verify state records ignored sessions with ignored: True
    state = json.loads(state_file.read_text(encoding="utf-8"))
    synced_map = state.get("synced_files", {})
    assert any("session_sens.jsonl" in k and v.get("ignored") is True for k, v in synced_map.items())
    assert any("session_sent.jsonl" in k and v.get("ignored") is True for k, v in synced_map.items())
    assert any("session_pub.jsonl" in k and v.get("ignored") is not True for k, v in synced_map.items())

