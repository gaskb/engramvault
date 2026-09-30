import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Union
from engramvault.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp


def _extract_chatgpt_messages(mapping: Dict[str, Any], current_node: str | None) -> List[NormalizedMessage]:
    """Extracts an ordered list of user and assistant messages from the mapping graph."""
    if not mapping:
        return []

    # If current_node is known, backtrack to root for the active conversation path
    path_nodes = []
    if current_node and current_node in mapping:
        curr = current_node
        visited = set()
        while curr and curr in mapping and curr not in visited:
            visited.add(curr)
            path_nodes.append(curr)
            curr = mapping[curr].get("parent")
        path_nodes.reverse()
    else:
        # Fallback: collect all nodes with messages and sort by create_time
        nodes_with_time = []
        for nid, node in mapping.items():
            msg = node.get("message")
            if msg and msg.get("create_time"):
                nodes_with_time.append((msg.get("create_time", 0), nid))
        nodes_with_time.sort()
        path_nodes = [nid for _, nid in nodes_with_time]

    messages: List[NormalizedMessage] = []
    for nid in path_nodes:
        node = mapping.get(nid, {})
        msg_obj = node.get("message")
        if not msg_obj:
            continue

        author = msg_obj.get("author", {})
        role = author.get("role")
        if role not in ("user", "assistant"):
            continue

        content_obj = msg_obj.get("content", {})
        parts = content_obj.get("parts", [])
        text_pieces = []
        for part in parts:
            if isinstance(part, str):
                text_pieces.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_pieces.append(part["text"])

        full_text = "\n".join(text_pieces).strip()
        if not full_text:
            continue

        ts = format_iso_timestamp(msg_obj.get("create_time"))
        messages.append(NormalizedMessage(
            role=role,
            text=full_text,
            ts=ts,
            metadata={"original_message_id": msg_obj.get("id")}
        ))

    return messages


def parse_chatgpt(source: Union[str, Path, bytes, List[Dict[str, Any]]]) -> List[NormalizedConversation]:
    """Parses a ChatGPT export file (JSON or ZIP), directory, or raw data."""
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
        conv_id = item.get("id") or item.get("conversation_id") or ""
        title = item.get("title") or "Senza titolo"
        create_time = item.get("create_time")
        date_iso = format_iso_timestamp(create_time)

        mapping = item.get("mapping", {})
        current_node = item.get("current_node")
        messages = _extract_chatgpt_messages(mapping, current_node)

        if not messages:
            # Fallback if messages are already in flat array (e.g. pre-parsed)
            if "messages" in item and isinstance(item["messages"], list):
                for m in item["messages"]:
                    r = m.get("role")
                    t = m.get("text") or m.get("content") or ""
                    if r in ("user", "assistant") and t:
                        messages.append(NormalizedMessage(role=r, text=t, ts=format_iso_timestamp(m.get("ts"))))

        if not messages:
            continue

        conversations.append(NormalizedConversation(
            assistant="chatgpt",
            title=title,
            date=date_iso,
            external_ref=conv_id,
            messages=messages,
            metadata={"model": item.get("default_model_slug")}
        ))

    return conversations
