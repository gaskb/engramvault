import json
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from memoria.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp


def _clean_user_request(raw_text: str) -> str:
    """Extracts human prompt text from Antigravity user input, removing metadata XML blocks."""
    if not raw_text:
        return ""

    match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", raw_text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = raw_text

    # Remove system and metadata tags that Antigravity appends to user turns
    text = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", text, flags=re.DOTALL)
    text = re.sub(r"<USER_INSTRUCTIONS>.*?</USER_INSTRUCTIONS>", "", text, flags=re.DOTALL)
    text = re.sub(r"<TASK_DETAILS>.*?</TASK_DETAILS>", "", text, flags=re.DOTALL)

    return text.strip()


def _find_antigravity_db(start_path: Path) -> Optional[Path]:
    """Looks for conversation_summaries.db nearby or in default ~/.gemini/antigravity location."""
    # Check parent directories
    curr = start_path.resolve()
    for _ in range(5):
        cand = curr / "conversation_summaries.db"
        if cand.is_file():
            return cand
        if curr.parent == curr:
            break
        curr = curr.parent

    # Default user directory
    default_db = Path.home() / ".gemini" / "antigravity" / "conversation_summaries.db"
    if default_db.is_file():
        return default_db

    return None


def _load_db_summaries(db_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Loads metadata index from conversation_summaries.db if present."""
    if not db_path or not db_path.is_file():
        return {}

    summaries: Dict[str, Dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        for row in cur.execute(
            "SELECT conversation_id, title, last_modified_time, workspace_uris FROM conversation_summaries"
        ):
            summaries[row[0]] = {
                "title": row[1] or "",
                "last_modified_time": row[2] or "",
                "workspace_uris": row[3] or "",
            }
        conn.close()
    except Exception:
        pass
    return summaries


def parse_antigravity_session(
    file_path: Path,
    summaries_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[NormalizedConversation]:
    """Parses a single Antigravity session transcript file (transcript_full.jsonl or transcript.jsonl)."""
    if not file_path.is_file():
        return None

    # Determine conversation_id: e.g. brain/<conv_id>/.system_generated/logs/transcript*.jsonl
    conv_id = ""
    for parent in file_path.parents:
        if parent.parent.name == "brain":
            conv_id = parent.name
            break
    if not conv_id:
        conv_id = file_path.stem

    messages: List[NormalizedMessage] = []
    first_ts = None
    last_ts = None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                step_type = entry.get("type")
                created_at = entry.get("created_at")
                iso_ts = format_iso_timestamp(created_at)

                if step_type == "USER_INPUT":
                    content = entry.get("content", "")
                    clean_text = _clean_user_request(content)
                    if clean_text:
                        if not first_ts:
                            first_ts = iso_ts
                        last_ts = iso_ts
                        messages.append(
                            NormalizedMessage(
                                role="user",
                                text=clean_text,
                                ts=iso_ts,
                                metadata={"step_index": entry.get("step_index")},
                            )
                        )
                elif step_type == "PLANNER_RESPONSE":
                    content = entry.get("content", "")
                    if content and content.strip():
                        if not first_ts:
                            first_ts = iso_ts
                        last_ts = iso_ts
                        messages.append(
                            NormalizedMessage(
                                role="assistant",
                                text=content.strip(),
                                ts=iso_ts,
                                metadata={"step_index": entry.get("step_index")},
                            )
                        )
    except Exception:
        return None

    if not messages:
        return None

    # Resolve metadata from summaries map if available
    db_meta = (summaries_map or {}).get(conv_id, {})
    title = db_meta.get("title")
    if not title:
        first_user = next((m.text for m in messages if m.role == "user"), "")
        first_line = first_user.split("\n")[0].strip()
        title = first_line[:80] if first_line else f"Sessione Antigravity {conv_id[:8]}"

    conv_date = db_meta.get("last_modified_time") or last_ts or first_ts or format_iso_timestamp(file_path.stat().st_mtime)

    metadata: Dict[str, Any] = {
        "file_path": str(file_path),
        "conversation_id": conv_id,
    }
    if db_meta.get("workspace_uris"):
        metadata["workspace_uris"] = db_meta["workspace_uris"]

    return NormalizedConversation(
        assistant="antigravity",
        title=title,
        date=format_iso_timestamp(conv_date),
        external_ref=conv_id,
        messages=messages,
        metadata=metadata,
    )


def parse_antigravity(source: Union[str, Path]) -> List[NormalizedConversation]:
    """
    Parses Antigravity data from:
    1. Single transcript file (.jsonl)
    2. Directory (e.g. ~/.gemini/antigravity or ~/.gemini/antigravity/brain)
    3. Zip archive containing Antigravity session files
    """
    p = Path(source)
    if not p.exists():
        return []

    if p.is_file() and p.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tmp_dir)
            return parse_antigravity(Path(tmp_dir))

    db_path = _find_antigravity_db(p)
    summaries_map = _load_db_summaries(db_path)

    conversations: List[NormalizedConversation] = []

    if p.is_file():
        conv = parse_antigravity_session(p, summaries_map)
        if conv:
            conversations.append(conv)
        return conversations

    # Directory scanning
    brain_dir = p / "brain" if (p / "brain").is_dir() else p

    # Find conversation directories
    # Look for brain/<conv_id> directories or direct transcript files
    handled_convs = set()

    # Search for transcript_full.jsonl or transcript.jsonl
    for candidate in sorted(brain_dir.glob("**/transcript_full.jsonl")):
        conv = parse_antigravity_session(candidate, summaries_map)
        if conv:
            conversations.append(conv)
            handled_convs.add(conv.external_ref)

    # Fallback to transcript.jsonl for any not already parsed via transcript_full
    for candidate in sorted(brain_dir.glob("**/transcript.jsonl")):
        cid = ""
        for parent in candidate.parents:
            if parent.parent.name == "brain":
                cid = parent.name
                break
        if cid and cid in handled_convs:
            continue

        conv = parse_antigravity_session(candidate, summaries_map)
        if conv and conv.external_ref not in handled_convs:
            conversations.append(conv)
            handled_convs.add(conv.external_ref)

    # If no standard brain structure was found, try any .jsonl in the folder
    if not conversations:
        for jsonl_file in sorted(brain_dir.glob("**/*.jsonl")):
            conv = parse_antigravity_session(jsonl_file, summaries_map)
            if conv and conv.external_ref not in handled_convs:
                conversations.append(conv)
                handled_convs.add(conv.external_ref)

    return conversations
