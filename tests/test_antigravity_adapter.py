import json
import sqlite3
import zipfile
from pathlib import Path

from engramvault.adapters.antigravity import _clean_user_request, parse_antigravity
from engramvault.adapters.registry import detect_provider, parse_source


def test_clean_user_request():
    raw = """<USER_REQUEST>
Quali sono le migliori audiocassette rock?
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-22T07:00:00+02:00.
</ADDITIONAL_METADATA>"""
    cleaned = _clean_user_request(raw)
    assert cleaned == "Quali sono le migliori audiocassette rock?"


def test_clean_user_request_without_tags():
    raw = "Prompt semplice senza tag speciali."
    assert _clean_user_request(raw) == "Prompt semplice senza tag speciali."


def test_parse_single_session_with_db(tmp_path: Path):
    base = tmp_path
    db_path = base / "conversation_summaries.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE conversation_summaries ("
        "conversation_id text PRIMARY KEY, title text, last_modified_time datetime, workspace_uris text)"
    )
    cid = "test-conv-uuid-1234"
    conn.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)",
        (cid, "Audiocassette Vintage Rock", "2026-09-22T08:00:00Z", "file:///homelab"),
    )
    conn.commit()
    conn.close()

    logs_dir = base / "brain" / cid / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True)
    transcript = logs_dir / "transcript_full.jsonl"

    lines = [
        {
            "step_index": 1,
            "type": "USER_INPUT",
            "created_at": "2026-09-22T08:00:00Z",
            "content": "<USER_REQUEST>Consigliami 3 nastri</USER_REQUEST>",
        },
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-09-22T08:00:05Z",
            "tool_calls": [{"name": "search"}],
        },
        {
            "step_index": 3,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-09-22T08:00:10Z",
            "content": "1. TDK SA-X\n2. Maxell XLII\n3. Sony HF",
        },
    ]
    with open(transcript, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")

    convs = parse_antigravity(base)
    assert len(convs) == 1
    c = convs[0]
    assert c.assistant == "antigravity"
    assert c.title == "Audiocassette Vintage Rock"
    assert c.external_ref == cid
    assert len(c.messages) == 2
    assert c.messages[0].role == "user"
    assert c.messages[0].text == "Consigliami 3 nastri"
    assert c.messages[1].role == "assistant"
    assert "TDK SA-X" in c.messages[1].text


def test_parse_from_zip_archive(tmp_path: Path):
    base = tmp_path
    zip_path = base / "antigravity_export.zip"
    cid = "zip-conv-999"

    lines = [
        {
            "step_index": 1,
            "type": "USER_INPUT",
            "created_at": "2026-09-20T10:00:00Z",
            "content": "<USER_REQUEST>Test da file zip</USER_REQUEST>",
        },
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-09-20T10:00:05Z",
            "content": "Risposta estratta da zip.",
        },
    ]
    transcript_content = "\n".join(json.dumps(item) for item in lines)

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"brain/{cid}/.system_generated/logs/transcript_full.jsonl", transcript_content)

    convs = parse_antigravity(zip_path)
    assert len(convs) == 1
    assert convs[0].assistant == "antigravity"
    assert convs[0].external_ref == cid
    assert convs[0].title == "Test da file zip"
    assert len(convs[0].messages) == 2


def test_provider_detection_and_parse_source(tmp_path: Path):
    file_path = tmp_path / "transcript.jsonl"
    lines = [
        {"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>Ciao Antigravity</USER_REQUEST>"},
        {"step_index": 2, "type": "PLANNER_RESPONSE", "content": "Ciao! Sono pronto ad aiutarti."},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item) + "\n")

    prov = detect_provider(file_path)
    assert prov == "antigravity"

    convs = parse_source(file_path)
    assert len(convs) == 1
    assert convs[0].assistant == "antigravity"

