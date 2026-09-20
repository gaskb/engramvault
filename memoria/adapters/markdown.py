import re
from pathlib import Path
from typing import List, Union
from memoria.adapters.base import NormalizedConversation, NormalizedMessage, format_iso_timestamp


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    metadata = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip().lower()] = v.strip().strip("\"'")
    return metadata, body


def parse_markdown_file(file_path: Path) -> NormalizedConversation | None:
    """Parses a markdown chat log with YAML frontmatter or role headings."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    metadata, body = _parse_frontmatter(content)
    title = metadata.get("title") or file_path.stem.replace("-", " ").replace("_", " ").title()
    date_val = metadata.get("date") or metadata.get("created_at") or format_iso_timestamp(file_path.stat().st_mtime)
    provider = metadata.get("provider") or metadata.get("assistant") or "markdown"

    # Pattern for splitting user and assistant turns:
    # Matches headings like:
    # ### User: / ### Assistant: / ### Claude: / ### ChatGPT: / **User:** / **Assistant:**
    pattern = re.compile(
        r"(?:^|\n)(?:###|\*\*|##)\s*(User|Human|Assistant|Claude|ChatGPT|Gemini|Model)[\*:]?\s*\n",
        re.IGNORECASE
    )

    splits = pattern.split(body)
    messages: List[NormalizedMessage] = []

    if len(splits) > 1:
        # splits[0] is preamble before first speaker
        # splits[1] is speaker, splits[2] is text, splits[3] is speaker...
        for i in range(1, len(splits), 2):
            raw_speaker = splits[i].strip().lower()
            text = splits[i+1].strip() if i+1 < len(splits) else ""
            if not text:
                continue

            role = "user" if raw_speaker in ("user", "human") else "assistant"
            messages.append(NormalizedMessage(
                role=role,
                text=text,
                ts=format_iso_timestamp(date_val)
            ))
    else:
        # If no explicit headings, treat whole content as a single document/turn
        messages.append(NormalizedMessage(
            role="user",
            text=body.strip(),
            ts=format_iso_timestamp(date_val)
        ))

    if not messages:
        return None

    return NormalizedConversation(
        assistant=provider,
        title=title,
        date=format_iso_timestamp(date_val),
        external_ref=file_path.name,
        messages=messages,
        metadata=metadata
    )


def parse_markdown(source: Union[str, Path]) -> List[NormalizedConversation]:
    """Parses single markdown file or directory of markdown files."""
    p = Path(source)
    results = []
    if p.is_file() and p.suffix.lower() in (".md", ".markdown"):
        c = parse_markdown_file(p)
        if c:
            results.append(c)
    elif p.is_dir():
        for f in p.glob("**/*.md"):
            c = parse_markdown_file(f)
            if c:
                results.append(c)
    return results
