import json
from pathlib import Path

from memoria.adapters.chatgpt import parse_chatgpt
from memoria.adapters.claude import parse_claude
from memoria.adapters.claude_code import parse_claude_code
from memoria.adapters.markdown import parse_markdown


def test_chatgpt_adapter():
    data = [
        {
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
                        "create_time": 1715000000.0,
                    },
                },
                "node2": {
                    "id": "node2",
                    "parent": "node1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Ciao! Come posso aiutarti?"]},
                        "create_time": 1715000010.0,
                    },
                },
            },
        }
    ]
    convs = parse_chatgpt(data)
    assert len(convs) == 1
    assert convs[0].title == "Chat di prova"
    assert convs[0].assistant == "chatgpt"
    assert len(convs[0].messages) == 2
    assert convs[0].messages[0].role == "user"
    assert convs[0].messages[0].text == "Ciao ChatGPT"
    assert convs[0].messages[1].role == "assistant"


def test_claude_adapter():
    data = [
        {
            "uuid": "claude-uuid-1",
            "name": "Discussione Audio",
            "created_at": "2024-05-01T12:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "Parlami dei registratori", "created_at": "2024-05-01T12:00:00Z"},
                {"sender": "assistant", "text": "I registratori a nastro...", "created_at": "2024-05-01T12:00:30Z"},
            ],
        }
    ]
    convs = parse_claude(data)
    assert len(convs) == 1
    assert convs[0].assistant == "claude"
    assert len(convs[0].messages) == 2
    assert convs[0].messages[0].role == "user"


def test_claude_code_adapter(tmp_path: Path):
    session_file = tmp_path / "session_123.jsonl"
    lines = [
        {"type": "user", "message": {"content": "Iniziamo il refactoring"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Analizzo il codice..."}]}},
    ]
    with open(session_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    convs = parse_claude_code(session_file)
    assert len(convs) == 1
    assert convs[0].assistant == "claude_code"
    assert len(convs[0].messages) == 2
    assert "refactoring" in convs[0].messages[0].text.lower()


def test_markdown_adapter(tmp_path: Path):
    md_file = tmp_path / "test_chat.md"
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
        encoding="utf-8",
    )
    convs = parse_markdown(md_file)
    assert len(convs) == 1
    assert convs[0].title == "Appunti e Chat"
    assert convs[0].assistant == "custom_ai"
    assert len(convs[0].messages) == 2

