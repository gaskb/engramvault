import json
import time
from pathlib import Path
import pytest
from engramvault.db.engine import Database
from engramvault.ingest.pipeline import ingest_source
from engramvault.ingest.watcher import DropFolderWatcher


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_vault.db"
    return Database(db_file)


def test_ingest_source(temp_db, tmp_path):
    # Create sample normalized JSON
    conv_data = [{
        "assistant": "claude",
        "title": "Chat di Test",
        "date": "2024-05-10T12:00:00Z",
        "external_ref": "test-ref-1",
        "messages": [
            {"role": "user", "text": "Ho una interfaccia Audient iD22", "ts": "2024-05-10T12:00:00Z"},
            {"role": "assistant", "text": "Ottima interfaccia.", "ts": "2024-05-10T12:01:00Z"},
        ]
    }]
    test_file = tmp_path / "sample.json"
    test_file.write_text(json.dumps(conv_data), encoding="utf-8")

    res = ingest_source(test_file, temp_db, provider="normalized")
    assert res["status"] == "success"
    assert res["imported"] == 1
    assert res["skipped"] == 0

    # Ingest again - should skip
    res2 = ingest_source(test_file, temp_db, provider="normalized")
    assert res2["imported"] == 0
    assert res2["skipped"] == 1


def test_drop_folder_watcher(temp_db, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    watcher = DropFolderWatcher(temp_db, incoming_dir=incoming, poll_interval=1)

    # 1. Add valid JSON file
    conv_data = [{
        "assistant": "chatgpt",
        "title": "Conversazione Drop",
        "date": "2024-06-01T10:00:00Z",
        "external_ref": "drop-ref-1",
        "messages": [
            {"role": "user", "text": "Uso Docker su Debian", "ts": "2024-06-01T10:00:00Z"},
            {"role": "assistant", "text": "Perfetto per i container.", "ts": "2024-06-01T10:01:00Z"},
        ]
    }]
    valid_file = incoming / "export.json"
    valid_file.write_text(json.dumps(conv_data), encoding="utf-8")

    # Set last_sizes to simulate file was written > 1s ago
    watcher._last_sizes[str(valid_file)] = (valid_file.stat().st_size, time.time() - 2.0)

    # 2. Add transient file (should be ignored)
    tmp_file = incoming / "download.zip.tmp"
    tmp_file.write_text("partial data")

    # 3. Add unsupported file (should be ignored)
    iso_file = incoming / "image.iso"
    iso_file.write_text("random")

    # Run scan
    results = watcher.scan_once()

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["imported"] == 1

    # Verify processed file was deleted from incoming (as per Issue #4)
    assert not valid_file.exists()
    # Verify transient and unsupported files were NOT deleted
    assert tmp_file.exists()
    assert iso_file.exists()

    # Verify conversation is in DB
    stats = temp_db.get_stats()
    assert stats["conversations"] == 1


def test_drop_folder_corrupted_file(temp_db, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    watcher = DropFolderWatcher(temp_db, incoming_dir=incoming)

    corrupted_file = incoming / "corrupted.json"
    corrupted_file.write_text("{invalid json non parseable")
    watcher._last_sizes[str(corrupted_file)] = (corrupted_file.stat().st_size, time.time() - 2.0)

    results = watcher.scan_once()
    assert len(results) == 1
    assert results[0]["status"] == "error"

    # Original corrupted file should be renamed to .failed
    assert not corrupted_file.exists()
    assert (incoming / "corrupted.json.failed").exists()
    assert (incoming / "corrupted.json.err.txt").exists()
