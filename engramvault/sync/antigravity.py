import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from engramvault.config import DEFAULT_DATA_DIR
from engramvault.db.engine import Database
from engramvault.ingest.pipeline import ingest_source
from engramvault.sync.claude_code import post_multipart_file, load_sync_state, save_sync_state
from engramvault.sync.filter import SyncFilter

logger = logging.getLogger("engramvault.sync.antigravity")

DEFAULT_ANTIGRAVITY_DIR = Path.home() / ".gemini" / "antigravity"
DEFAULT_ANTIGRAVITY_STATE_FILE = DEFAULT_DATA_DIR / "antigravity_sync_state.json"


class AntigravitySyncer:
    """
    Scans ~/.gemini/antigravity/ and synchronizes new or updated session transcripts
    to EngramVault (local database or remote instance), respecting exclusion rules.
    """

    def __init__(
        self,
        antigravity_dir: Optional[Path | str] = None,
        state_file: Optional[Path | str] = None,
        remote_url: Optional[str] = None,
        db: Optional[Database] = None,
        api_token: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
        ignore_file: Optional[Path | str] = None,
    ):
        self.antigravity_dir = Path(antigravity_dir or DEFAULT_ANTIGRAVITY_DIR)
        self.state_file = Path(state_file or DEFAULT_ANTIGRAVITY_STATE_FILE)
        self.remote_url = (remote_url or "").rstrip("/")
        self.api_token = api_token or os.environ.get("MEMORIA_API_TOKEN")
        self.db = db
        self.sync_filter = SyncFilter(patterns=exclude_patterns, ignore_file=ignore_file)

    def load_conversation_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Loads metadata index from conversation_summaries.db if present."""
        db_path = self.antigravity_dir / "conversation_summaries.db"
        if not db_path.is_file():
            return {}
        import sqlite3
        summaries: Dict[str, Dict[str, Any]] = {}
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            for row in cur.execute("SELECT * FROM conversation_summaries"):
                cid = row["conversation_id"] if "conversation_id" in row.keys() else ""
                if not cid:
                    continue
                summaries[cid] = {
                    "title": row["title"] if "title" in row.keys() and row["title"] else "",
                    "last_modified_time": row["last_modified_time"] if "last_modified_time" in row.keys() and row["last_modified_time"] else "",
                    "workspace_uris": row["workspace_uris"] if "workspace_uris" in row.keys() and row["workspace_uris"] else "",
                }
            conn.close()
        except Exception as e:
            logger.debug(f"Impossibile leggere conversation_summaries.db: {e}")
        return summaries

    def _is_session_ignored(
        self,
        cid: str,
        summaries: Dict[str, Dict[str, Any]],
        file_path: Path,
    ) -> bool:
        """Determines if an Antigravity session belongs to an excluded workspace or contains sentinel files."""
        meta = summaries.get(cid, {})
        title = meta.get("title", "")
        ws_uris = meta.get("workspace_uris", "")

        # 1. Check title against pattern
        if title and self.sync_filter.matches_pattern(title):
            return True

        # 2. Check workspace URIs
        if ws_uris:
            try:
                uri_list = json.loads(ws_uris) if ws_uris.startswith("[") else [ws_uris]
                for uri in uri_list:
                    if self.sync_filter.is_workspace_ignored(uri):
                        return True
            except Exception:
                if self.sync_filter.is_workspace_ignored(ws_uris):
                    return True

        # 3. Check for workspace info in transcript first lines if not in db
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx > 30:
                        break
                    if "workspace" in line.lower() or "file://" in line:
                        for token in line.split():
                            if token.startswith("file://") or token.startswith("/"):
                                clean_token = token.strip('"\' ,;[]')
                                if self.sync_filter.is_workspace_ignored(clean_token):
                                    return True
        except Exception:
            pass

        return False

    def scan_changed_files(self) -> List[Path]:
        """Returns list of transcript files that are new or have changed since last sync and not ignored."""
        if not self.antigravity_dir.exists():
            return []

        brain_dir = self.antigravity_dir / "brain" if (self.antigravity_dir / "brain").is_dir() else self.antigravity_dir

        state = load_sync_state(self.state_file)
        synced_map = state.get("synced_files", {})
        summaries = self.load_conversation_summaries()

        changed: List[Path] = []
        handled_convs = set()
        ignored_count = 0

        # Check transcript_full.jsonl first
        for file_path in sorted(brain_dir.glob("**/transcript_full.jsonl")):
            if not file_path.is_file():
                continue

            cid = ""
            for parent in file_path.parents:
                if parent.parent.name == "brain":
                    cid = parent.name
                    break
            if not cid:
                cid = file_path.stem
            handled_convs.add(cid)

            rel_key = str(file_path.relative_to(self.antigravity_dir))
            stat = file_path.stat()
            prev = synced_map.get(rel_key)

            # Check if ignored by filter
            if self._is_session_ignored(cid, summaries, file_path):
                ignored_count += 1
                if not prev or prev.get("mtime") != stat.st_mtime or prev.get("size") != stat.st_size:
                    synced_map[rel_key] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "ignored": True,
                    }
                continue

            if not prev or prev.get("mtime") != stat.st_mtime or prev.get("size") != stat.st_size:
                changed.append(file_path)

        # Fallback to transcript.jsonl if full is not present
        for file_path in sorted(brain_dir.glob("**/transcript.jsonl")):
            if not file_path.is_file():
                continue

            cid = ""
            for parent in file_path.parents:
                if parent.parent.name == "brain":
                    cid = parent.name
                    break
            if cid and cid in handled_convs:
                continue

            rel_key = str(file_path.relative_to(self.antigravity_dir))
            stat = file_path.stat()
            prev = synced_map.get(rel_key)

            # Check if ignored by filter
            if self._is_session_ignored(cid, summaries, file_path):
                ignored_count += 1
                if not prev or prev.get("mtime") != stat.st_mtime or prev.get("size") != stat.st_size:
                    synced_map[rel_key] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "ignored": True,
                    }
                continue

            if not prev or prev.get("mtime") != stat.st_mtime or prev.get("size") != stat.st_size:
                changed.append(file_path)

        if ignored_count > 0:
            save_sync_state(self.state_file, state)

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
                "files": [str(f.relative_to(self.antigravity_dir)) for f in changed_files[:20]],
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
        db_summaries_file = self.antigravity_dir / "conversation_summaries.db"

        # Process in batches
        for i in range(0, len(changed_files), max_batch_size):
            batch = changed_files[i : i + max_batch_size]

            if self.remote_url:
                # Pack batch into temporary zip file preserving structure and summaries db
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
                    tmp_zip_path = Path(tmp_zip.name)

                try:
                    with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        if db_summaries_file.is_file():
                            zf.write(db_summaries_file, arcname="conversation_summaries.db")
                        for f in batch:
                            rel_name = str(f.relative_to(self.antigravity_dir))
                            zf.write(f, arcname=rel_name)

                    endpoint = f"{self.remote_url}/api/ingest"
                    resp = post_multipart_file(
                        endpoint,
                        tmp_zip_path,
                        form_fields={"provider": "antigravity"},
                        api_token=self.api_token,
                    )

                    if resp.get("status") == "success":
                        for f in batch:
                            rel_key = str(f.relative_to(self.antigravity_dir))
                            stat = f.stat()
                            synced_map[rel_key] = {
                                "mtime": stat.st_mtime,
                                "size": stat.st_size,
                            }
                            synced_count += 1
                    else:
                        error_count += len(batch)
                except Exception as e:
                    logger.error(f"Errore durante l'invio del batch Antigravity a {self.remote_url}: {e}")
                    error_count += len(batch)
                finally:
                    tmp_zip_path.unlink(missing_ok=True)

            elif self.db:
                # Local ingest directly into database
                for f in batch:
                    try:
                        ingest_source(f, self.db, provider="antigravity")
                        rel_key = str(f.relative_to(self.antigravity_dir))
                        stat = f.stat()
                        synced_map[rel_key] = {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                        }
                        synced_count += 1
                    except Exception as e:
                        logger.error(f"Errore durante l'ingest locale Antigravity di {f}: {e}")
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
