import json
import zipfile
from pathlib import Path
from typing import List, Union
from memoria.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp
from memoria.adapters.chatgpt import parse_chatgpt
from memoria.adapters.claude import parse_claude
from memoria.adapters.claude_code import parse_claude_code
from memoria.adapters.gemini import parse_gemini
from memoria.adapters.markdown import parse_markdown


def detect_provider(path: Path) -> str:
    """Detects provider type from file or directory structure."""
    if path.is_dir():
        if list(path.glob("**/*.jsonl")):
            return "claude_code"
        if list(path.glob("**/*.md")):
            return "markdown"
        return "unknown"

    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return "markdown"
    if suffix == ".jsonl":
        return "claude_code"

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path, "r") as z:
                names = z.namelist()
                if any(n.endswith(".jsonl") for n in names):
                    return "claude_code"
                if any("conversations.json" in n for n in names):
                    # Check snippet of conversations.json
                    for n in names:
                        if n.endswith("conversations.json"):
                            with z.open(n) as f:
                                prefix = f.read(4096).decode("utf-8", errors="ignore")
                                if '"mapping"' in prefix:
                                    return "chatgpt"
                                if '"chat_messages"' in prefix:
                                    return "claude"
                    return "chatgpt"  # default for conversations.json
        except Exception:
            return "unknown"

    if suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                prefix = f.read(4096)
                if '"mapping"' in prefix:
                    return "chatgpt"
                if '"chat_messages"' in prefix or ('"uuid"' in prefix and '"sender"' in prefix):
                    return "claude"
                if '"assistant"' in prefix and '"external_ref"' in prefix:
                    return "normalized"
                if "gemini" in path.name.lower() or "takeout" in path.name.lower():
                    return "gemini"
        except Exception:
            return "unknown"

    return "unknown"


def parse_normalized_json(source: Union[str, Path]) -> List[NormalizedConversation]:
    """Parses already normalized conversation JSON file."""
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []

    results = []
    for item in data:
        messages = []
        for m in item.get("messages", []):
            messages.append(NormalizedMessage(
                role=m.get("role", "user"),
                text=m.get("text") or m.get("content") or "",
                ts=format_iso_timestamp(m.get("ts")),
                metadata=m.get("metadata", {})
            ))
        results.append(NormalizedConversation(
            assistant=item.get("assistant", "unknown"),
            title=item.get("title", "Senza titolo"),
            date=format_iso_timestamp(item.get("date")),
            external_ref=item.get("external_ref", ""),
            messages=messages,
            metadata=item.get("metadata", {})
        ))
    return results


def parse_source(source: Union[str, Path], provider: str = "auto") -> List[NormalizedConversation]:
    """Universal dispatcher for importing conversations from any provider."""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Percorso non trovato: {source}")

    if provider == "auto" or not provider:
        provider = detect_provider(p)

    if provider == "chatgpt":
        return parse_chatgpt(p)
    elif provider == "claude":
        return parse_claude(p)
    elif provider == "claude_code":
        return parse_claude_code(p)
    elif provider == "gemini":
        return parse_gemini(p)
    elif provider == "markdown":
        return parse_markdown(p)
    elif provider == "normalized":
        return parse_normalized_json(p)
    else:
        # Try all parsers gracefully
        for parser in (parse_normalized_json, parse_chatgpt, parse_claude, parse_claude_code, parse_gemini, parse_markdown):
            try:
                res = parser(p)
                if res:
                    return res
            except Exception:
                continue

    return []
