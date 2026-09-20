from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import List, Dict, Any, Optional


@dataclass
class NormalizedMessage:
    role: str  # "user" or "assistant"
    text: str
    ts: str = ""  # ISO 8601 string
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "ts": self.ts,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedConversation:
    assistant: str  # "chatgpt", "claude", "claude_code", "gemini", "markdown"
    title: str
    date: str  # ISO 8601 string
    external_ref: str
    messages: List[NormalizedMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Computes a deterministic SHA256 hash for content-based deduplication."""
        hasher = hashlib.sha256()
        hasher.update(self.assistant.encode("utf-8"))
        hasher.update(self.external_ref.encode("utf-8"))
        hasher.update(self.title.encode("utf-8"))
        for msg in self.messages:
            hasher.update(msg.role.encode("utf-8"))
            hasher.update(msg.text.strip().encode("utf-8"))
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assistant": self.assistant,
            "title": self.title,
            "date": self.date,
            "external_ref": self.external_ref,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "content_hash": self.compute_hash(),
        }


def format_iso_timestamp(ts: Optional[float | int | str]) -> str:
    """Helper to convert unix timestamp or string to clean UTC ISO8601 string."""
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    if isinstance(ts, str):
        try:
            # check if it is numeric string
            f = float(ts)
            return datetime.fromtimestamp(f, tz=timezone.utc).isoformat()
        except ValueError:
            return ts
    return datetime.now(timezone.utc).isoformat()
