import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Union
from memoria.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp


def parse_gemini(source: Union[str, Path, bytes, List[Dict[str, Any]]]) -> List[NormalizedConversation]:
    """Parses Gemini exports (JSON, ZIP, or Takeout activity)."""
    raw_data = None

    if isinstance(source, list):
        raw_data = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p, "r") as z:
                    for name in z.namelist():
                        if name.endswith(".json") and not name.startswith("__MACOSX"):
                            with z.open(name) as f:
                                try:
                                    loaded = json.load(f)
                                    if isinstance(loaded, list):
                                        raw_data = loaded
                                        break
                                except Exception:
                                    continue
            elif p.suffix.lower() == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
    elif isinstance(source, bytes):
        raw_data = json.loads(source.decode("utf-8"))

    if not raw_data:
        return []

    conversations: List[NormalizedConversation] = []

    # Case A: Standard normalized or structured chat list
    if isinstance(raw_data, list):
        for idx, item in enumerate(raw_data):
            # Check if this is already an activity format or a conversation object
            title = item.get("title") or item.get("name") or f"Conversazione Gemini #{idx+1}"
            date_val = item.get("date") or item.get("time") or item.get("create_time") or item.get("created_at")
            conv_id = item.get("external_ref") or item.get("id") or f"gemini_{idx}"

            messages: List[NormalizedMessage] = []

            # Sub-case: item has 'messages' or 'turns'
            m_list = item.get("messages") or item.get("turns") or []
            if m_list:
                for m in m_list:
                    role = m.get("role") or m.get("author")
                    if role in ("model", "assistant", "gemini"):
                        role = "assistant"
                    elif role in ("user", "human"):
                        role = "user"
                    else:
                        continue

                    text = m.get("text") or m.get("content") or ""
                    if text.strip():
                        messages.append(NormalizedMessage(
                            role=role,
                            text=text.strip(),
                            ts=format_iso_timestamp(m.get("ts") or m.get("time"))
                        ))

            # Sub-case: Takeout Activity format (often contains 'header', 'title', 'subtitles')
            elif "header" in item and "title" in item:
                prompt_text = item.get("title", "")
                if prompt_text.startswith("Ha richiesto: ") or prompt_text.startswith("Prompt: "):
                    prompt_text = prompt_text.split(": ", 1)[-1]
                if prompt_text:
                    messages.append(NormalizedMessage(
                        role="user",
                        text=prompt_text,
                        ts=format_iso_timestamp(date_val)
                    ))
                # Note: Takeout often omits AI answers, but if present in description:
                description = item.get("description")
                if description:
                    messages.append(NormalizedMessage(
                        role="assistant",
                        text=description,
                        ts=format_iso_timestamp(date_val)
                    ))

            if messages:
                conversations.append(NormalizedConversation(
                    assistant="gemini",
                    title=title,
                    date=format_iso_timestamp(date_val),
                    external_ref=str(conv_id),
                    messages=messages,
                    metadata={"source": "gemini_export"}
                ))

    return conversations
