import json
from pathlib import Path
from typing import List, Dict, Any, Union
from memoria.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp


def _clean_claude_code_content(content: Any) -> str:
    """Extracts human-readable text from Claude Code message content, filtering out raw tool json noise."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                b_type = block.get("type")
                if b_type == "text":
                    text_parts.append(block.get("text", ""))
                elif b_type == "tool_use":
                    tool_name = block.get("name", "tool")
                    # We can keep a high-level summary of tool intent without the huge payload
                    if tool_name not in ("run_command", "view_file"):
                        text_parts.append(f"*[Azione: {tool_name}]*")
                # Skip tool_result (large terminal stdout, raw files) to avoid noisy tokens
        return "\n".join(text_parts).strip()

    return ""


def parse_claude_code_session(file_path: Path) -> NormalizedConversation | None:
    """Parses a single Claude Code session .jsonl file."""
    if not file_path.is_file():
        return None

    messages: List[NormalizedMessage] = []
    first_ts = None
    session_id = file_path.stem

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Session line format varies slightly across Claude Code versions
                role = None
                raw_content = None
                ts = None

                if "type" in entry and entry["type"] in ("user", "assistant"):
                    role = entry["type"]
                    msg_obj = entry.get("message", {})
                    raw_content = msg_obj.get("content") or entry.get("content")
                    ts = entry.get("timestamp") or entry.get("created_at")
                elif "role" in entry and entry["role"] in ("user", "assistant"):
                    role = entry["role"]
                    raw_content = entry.get("content") or entry.get("text")
                    ts = entry.get("timestamp") or entry.get("created_at")

                if not role or not raw_content:
                    continue

                clean_text = _clean_claude_code_content(raw_content)
                if not clean_text:
                    continue

                iso_ts = format_iso_timestamp(ts)
                if not first_ts:
                    first_ts = iso_ts

                messages.append(NormalizedMessage(
                    role=role,
                    text=clean_text,
                    ts=iso_ts,
                    metadata={"session_file": str(file_path.name)}
                ))
    except Exception:
        return None

    if not messages:
        return None

    # Derive title from the first user prompt or directory
    first_user_msg = next((m.text for m in messages if m.role == "user"), "")
    title = first_user_msg.split("\n")[0][:80].strip() or f"Sessione {session_id}"

    project_name = file_path.parent.name if file_path.parent else "default"

    return NormalizedConversation(
        assistant="claude_code",
        title=f"[{project_name}] {title}",
        date=first_ts or format_iso_timestamp(file_path.stat().st_mtime),
        external_ref=session_id,
        messages=messages,
        metadata={"project": project_name, "file_path": str(file_path)}
    )


def parse_claude_code(source: Union[str, Path]) -> List[NormalizedConversation]:
    """Parses Claude Code session file or directory (e.g. ~/.claude/projects)."""
    p = Path(source)
    conversations: List[NormalizedConversation] = []

    if p.is_file():
        conv = parse_claude_code_session(p)
        if conv:
            conversations.append(conv)
    elif p.is_dir():
        for jsonl_file in p.glob("**/*.jsonl"):
            conv = parse_claude_code_session(jsonl_file)
            if conv:
                conversations.append(conv)

    return conversations
