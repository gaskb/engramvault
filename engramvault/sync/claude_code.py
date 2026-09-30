import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from engramvault.config import DEFAULT_DATA_DIR
from engramvault.db.engine import Database
from engramvault.ingest.pipeline import ingest_source

logger = logging.getLogger("engramvault.sync")

DEFAULT_CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_STATE_FILE = DEFAULT_DATA_DIR / "claude_code_sync_state.json"


def load_sync_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "synced_files": {}}
    return {"version": 1, "synced_files": {}}


def save_sync_state(state_file: Path, state: Dict[str, Any]):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def post_multipart_file(
    url: str,
    file_path: Path,
    form_fields: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    api_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Posts a file to an HTTP endpoint using standard multipart/form-data."""
    boundary = f"----MemoriaBoundary{uuid.uuid4().hex}"
    form_fields = form_fields or {}

    body_parts: List[bytes] = []

    # Add text form fields
    for field_name, field_val in form_fields.items():
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode("utf-8")
        )
        body_parts.append(f"{field_val}\r\n".encode("utf-8"))

    # Add file field
    file_bytes = file_path.read_bytes()
    body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    body_parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    body_parts.append(file_bytes)
    body_parts.append(b"\r\n")
    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    payload = b"".join(body_parts)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(payload)),
    }
    token = api_token or os.environ.get("MEMORIA_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


class ClaudeCodeSyncer:
    """
    Scans ~/.claude/projects/ and synchronizes new or updated session files (.jsonl)
    to prevent losing chat histories when Claude Code deletes them after 30 days.
    """

    def __init__(
        self,
        projects_dir: Optional[Path | str] = None,
        state_file: Optional[Path | str] = None,
        remote_url: Optional[str] = None,
        db: Optional[Database] = None,
        api_token: Optional[str] = None,
    ):
        self.projects_dir = Path(projects_dir or DEFAULT_CLAUDE_PROJECTS_DIR)
        self.state_file = Path(state_file or DEFAULT_STATE_FILE)
        self.remote_url = (remote_url or "").rstrip("/")
        self.api_token = api_token or os.environ.get("MEMORIA_API_TOKEN")
        self.db = db

    def scan_changed_files(self) -> List[Path]:
        """Returns list of session files that are new or have changed since last sync."""
        if not self.projects_dir.exists():
            return []

        state = load_sync_state(self.state_file)
        synced_map = state.get("synced_files", {})

        changed: List[Path] = []
        for file_path in self.projects_dir.glob("**/*.jsonl"):
            if not file_path.is_file():
                continue

            rel_key = str(file_path.relative_to(self.projects_dir))
            stat = file_path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size

            prev = synced_map.get(rel_key)
            if not prev or prev.get("mtime") != current_mtime or prev.get("size") != current_size:
                changed.append(file_path)

        return sorted(changed, key=lambda f: f.stat().st_mtime)

    def sync(self, dry_run: bool = False, max_batch_size: int = 50) -> Dict[str, Any]:
        """
        Executes sync of changed files to either remote instance or local database.
        """
        changed_files = self.scan_changed_files()
        total_found = len(changed_files)

        if dry_run:
            return {
                "status": "dry_run",
                "changed_count": total_found,
                "files": [str(f.relative_to(self.projects_dir)) for f in changed_files[:20]],
            }

        if not changed_files:
            return {
                "status": "up_to_date",
                "changed_count": 0,
                "synced_count": 0,
            }

        state = load_sync_state(self.state_file)
        synced_map = state.setdefault("synced_files", {})

        synced_count = 0
        error_count = 0

        # Process in batches to keep archive payload manageable
        for i in range(0, len(changed_files), max_batch_size):
            batch = changed_files[i : i + max_batch_size]

            if self.remote_url:
                # Pack batch into temporary zip file preserving project relative path
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    tmp_zip_path = Path(tmp_zip.name)

                try:
                    with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for f in batch:
                            rel_name = str(f.relative_to(self.projects_dir))
                            zf.write(f, arcname=rel_name)

                    endpoint = f"{self.remote_url}/api/ingest"
                    resp = post_multipart_file(
                        endpoint,
                        tmp_zip_path,
                        form_fields={"provider": "claude_code"},
                        api_token=self.api_token,
                    )

                    if resp.get("status") == "success":
                        for f in batch:
                            rel_key = str(f.relative_to(self.projects_dir))
                            stat = f.stat()
                            synced_map[rel_key] = {
                                "mtime": stat.st_mtime,
                                "size": stat.st_size,
                            }
                            synced_count += 1
                    else:
                        error_count += len(batch)
                except Exception as e:
                    logger.error(f"Errore durante l'invio del batch a {self.remote_url}: {e}")
                    error_count += len(batch)
                finally:
                    tmp_zip_path.unlink(missing_ok=True)

            elif self.db:
                # Local ingest directly into database
                for f in batch:
                    try:
                        ingest_source(f, self.db, provider="claude_code")
                        rel_key = str(f.relative_to(self.projects_dir))
                        stat = f.stat()
                        synced_map[rel_key] = {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                        synced_count += 1
                    except Exception as e:
                        logger.error(f"Errore durante l'ingest locale di {f}: {e}")
                        error_count += 1
            else:
                raise ValueError("Né remote_url né db sono stati configurati per la sincronizzazione.")

        save_sync_state(self.state_file, state)

        return {
            "status": "success",
            "changed_count": total_found,
            "synced_count": synced_count,
            "error_count": error_count,
        }
