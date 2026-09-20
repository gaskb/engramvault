import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Union
from memoria.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp


def parse_claude(source: Union[str, Path, bytes, List[Dict[str, Any]]]) -> List[NormalizedConversation]:
    """Parses a Claude Web export (JSON or ZIP)."""
    raw_data = None

    if isinstance(source, list):
        raw_data = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p, "r") as z:
                    for name in z.namelist():
                        if name.endswith("conversations.json"):
                            with z.open(name) as f:
                                raw_data = json.load(f)
                            break
            elif p.suffix.lower() == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
    elif isinstance(source, bytes):
        raw_data = json.loads(source.decode("utf-8"))

    if not raw_data or not isinstance(raw_data, list):
        return []

    conversations: List[NormalizedConversation] = []
    for item in raw_data:
        conv_id = item.get("uuid") or item.get("id") or ""
        title = item.get("name") or item.get("title") or "Senza titolo"
        date_iso = format_iso_timestamp(item.get("created_at") or item.get("updated_at"))

        chat_messages = item.get("chat_messages") or item.get("messages") or []
        messages: List[NormalizedMessage] = []

        for m in chat_messages:
            sender = m.get("sender") or m.get("role")
            if sender == "human":
                role = "user"
            elif sender == "assistant":
                role = "assistant"
            elif sender in ("user", "assistant"):
                role = sender
            else:
                continue

            text = m.get("text") or m.get("content") or ""
            if isinstance(text, list):
                # Claude sometimes has list of content blocks
                text_parts = []
                for b in text:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif isinstance(b, str):
                        text_parts.append(b)
                text = "\n".join(text_parts).strip()

            text = text.strip()
            if not text:
                continue

            ts = format_iso_timestamp(m.get("created_at") or m.get("ts"))
            messages.append(NormalizedMessage(
                role=role,
                text=text,
                ts=ts,
                metadata={"original_message_id": m.get("uuid")}
            ))

        if not messages:
            continue

        conversations.append(NormalizedConversation(
            assistant="claude",
            title=title,
            date=date_iso,
            external_ref=conv_id,
            messages=messages,
            metadata={"account_id": item.get("account_id")}
        ))

    return conversations
