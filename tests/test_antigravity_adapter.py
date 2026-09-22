import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from memoria.adapters.antigravity import parse_antigravity, parse_antigravity_session, _clean_user_request
from memoria.adapters.registry import detect_provider, parse_source


class TestAntigravityAdapter(unittest.TestCase):
    def test_clean_user_request(self):
        raw = """<USER_REQUEST>
Quali sono le migliori audiocassette rock?
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-22T07:00:00+02:00.
</ADDITIONAL_METADATA>"""
        cleaned = _clean_user_request(raw)
        self.assertEqual(cleaned, "Quali sono le migliori audiocassette rock?")

    def test_clean_user_request_without_tags(self):
        raw = "Prompt semplice senza tag speciali."
        self.assertEqual(_clean_user_request(raw), "Prompt semplice senza tag speciali.")

    def test_parse_single_session_with_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            # Create SQLite db
            db_path = base / "conversation_summaries.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "CREATE TABLE conversation_summaries ("
                "conversation_id text PRIMARY KEY, title text, last_modified_time datetime, workspace_uris text)"
            )
            cid = "test-conv-uuid-1234"
            conn.execute(
                "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?)",
                (cid, "Audiocassette Vintage Rock", "2026-09-22T08:00:00Z", "file:///homelab")
            )
            conn.commit()
            conn.close()

            # Create brain directory structure
            logs_dir = base / "brain" / cid / ".system_generated" / "logs"
            logs_dir.mkdir(parents=True)
            transcript = logs_dir / "transcript_full.jsonl"

            lines = [
                {"step_index": 1, "type": "USER_INPUT", "created_at": "2026-09-22T08:00:00Z", "content": "<USER_REQUEST>Consigliami 3 nastri</USER_REQUEST>"},
                {"step_index": 2, "type": "PLANNER_RESPONSE", "created_at": "2026-09-22T08:00:05Z", "tool_calls": [{"name": "search"}]},
                {"step_index": 3, "type": "PLANNER_RESPONSE", "created_at": "2026-09-22T08:00:10Z", "content": "1. TDK SA-X\n2. Maxell XLII\n3. Sony HF"},
            ]
            with open(transcript, "w", encoding="utf-8") as f:
                for item in lines:
                    f.write(json.dumps(item) + "\n")

            convs = parse_antigravity(base)
            self.assertEqual(len(convs), 1)
            c = convs[0]
            self.assertEqual(c.assistant, "antigravity")
            self.assertEqual(c.title, "Audiocassette Vintage Rock")
            self.assertEqual(c.external_ref, cid)
            self.assertEqual(len(c.messages), 2)
            self.assertEqual(c.messages[0].role, "user")
            self.assertEqual(c.messages[0].text, "Consigliami 3 nastri")
            self.assertEqual(c.messages[1].role, "assistant")
            self.assertIn("TDK SA-X", c.messages[1].text)

    def test_parse_from_zip_archive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            zip_path = base / "antigravity_export.zip"
            cid = "zip-conv-999"

            lines = [
                {"step_index": 1, "type": "USER_INPUT", "created_at": "2026-09-20T10:00:00Z", "content": "<USER_REQUEST>Test da file zip</USER_REQUEST>"},
                {"step_index": 2, "type": "PLANNER_RESPONSE", "created_at": "2026-09-20T10:00:05Z", "content": "Risposta estratta da zip."},
            ]
            transcript_content = "\n".join(json.dumps(l) for l in lines)

            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(f"brain/{cid}/.system_generated/logs/transcript_full.jsonl", transcript_content)

            convs = parse_antigravity(zip_path)
            self.assertEqual(len(convs), 1)
            self.assertEqual(convs[0].assistant, "antigravity")
            self.assertEqual(convs[0].external_ref, cid)
            self.assertEqual(convs[0].title, "Test da file zip")
            self.assertEqual(len(convs[0].messages), 2)

    def test_provider_detection_and_parse_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "transcript.jsonl"
            lines = [
                {"step_index": 1, "type": "USER_INPUT", "content": "<USER_REQUEST>Ciao Antigravity</USER_REQUEST>"},
                {"step_index": 2, "type": "PLANNER_RESPONSE", "content": "Ciao! Sono pronto ad aiutarti."},
            ]
            with open(file_path, "w", encoding="utf-8") as f:
                for l in lines:
                    f.write(json.dumps(l) + "\n")

            prov = detect_provider(file_path)
            self.assertEqual(prov, "antigravity")

            convs = parse_source(file_path)
            self.assertEqual(len(convs), 1)
            self.assertEqual(convs[0].assistant, "antigravity")


if __name__ == "__main__":
    unittest.main()
