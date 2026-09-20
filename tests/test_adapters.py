import json
import tempfile
import unittest
from pathlib import Path
from memoria.adapters.chatgpt import parse_chatgpt
from memoria.adapters.claude import parse_claude
from memoria.adapters.claude_code import parse_claude_code
from memoria.adapters.gemini import parse_gemini
from memoria.adapters.markdown import parse_markdown
from memoria.adapters.registry import detect_provider, parse_source


class TestAdapters(unittest.TestCase):
    def test_chatgpt_adapter(self):
        data = [{
            "id": "conv-123",
            "title": "Chat di prova",
            "create_time": 1715000000.0,
            "current_node": "node2",
            "mapping": {
                "node1": {
                    "id": "node1",
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Ciao ChatGPT"]},
                        "create_time": 1715000000.0
                    }
                },
                "node2": {
                    "id": "node2",
                    "parent": "node1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Ciao! Come posso aiutarti?"]},
                        "create_time": 1715000010.0
                    }
                }
            }
        }]
        convs = parse_chatgpt(data)
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].title, "Chat di prova")
        self.assertEqual(convs[0].assistant, "chatgpt")
        self.assertEqual(len(convs[0].messages), 2)
        self.assertEqual(convs[0].messages[0].role, "user")
        self.assertEqual(convs[0].messages[0].text, "Ciao ChatGPT")
        self.assertEqual(convs[0].messages[1].role, "assistant")

    def test_claude_adapter(self):
        data = [{
            "uuid": "claude-uuid-1",
            "name": "Discussione Audio",
            "created_at": "2024-05-01T12:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "Parlami dei registratori", "created_at": "2024-05-01T12:00:00Z"},
                {"sender": "assistant", "text": "I registratori a nastro...", "created_at": "2024-05-01T12:00:30Z"}
            ]
        }]
        convs = parse_claude(data)
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].assistant, "claude")
        self.assertEqual(len(convs[0].messages), 2)
        self.assertEqual(convs[0].messages[0].role, "user")

    def test_claude_code_adapter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_file = Path(tmp_dir) / "session_123.jsonl"
            lines = [
                {"type": "user", "message": {"content": "Iniziamo il refactoring"}},
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "Analizzo il codice..."}]}}
            ]
            with open(session_file, "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")

            convs = parse_claude_code(session_file)
            self.assertEqual(len(convs), 1)
            self.assertEqual(convs[0].assistant, "claude_code")
            self.assertEqual(len(convs[0].messages), 2)
            self.assertIn("refactoring", convs[0].messages[0].text.lower())

    def test_markdown_adapter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_file = Path(tmp_dir) / "test_chat.md"
            md_file.write_text(
                "---\n"
                "title: Appunti e Chat\n"
                "provider: custom_ai\n"
                "date: 2024-06-01T10:00:00Z\n"
                "---\n\n"
                "### User:\n"
                "Ho un dubbio sull'impasto.\n\n"
                "### Assistant:\n"
                "Ecco la risposta sull'impasto.\n",
                encoding="utf-8"
            )
            convs = parse_markdown(md_file)
            self.assertEqual(len(convs), 1)
            self.assertEqual(convs[0].title, "Appunti e Chat")
            self.assertEqual(convs[0].assistant, "custom_ai")
            self.assertEqual(len(convs[0].messages), 2)


if __name__ == "__main__":
    unittest.main()
