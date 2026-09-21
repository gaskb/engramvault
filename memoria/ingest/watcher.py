import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from memoria.config import DEFAULT_INCOMING_DIR, WATCH_INTERVAL
from memoria.db.engine import Database
from memoria.ingest.pipeline import ingest_source

logger = logging.getLogger("memoria.watcher")

SUPPORTED_EXTENSIONS = {".zip", ".tar.gz", ".tgz", ".json", ".jsonl", ".md", ".txt"}
TRANSIENT_EXTENSIONS = {".tmp", ".part", ".crdownload", ".uploading", ".swp"}


class DropFolderWatcher:
    """
    Monitors an incoming drop directory, automatically ingests supported archives
    and exports, and cleans up processed files upon completion.
    """

    def __init__(
        self,
        db: Database,
        incoming_dir: Optional[Path | str] = None,
        poll_interval: int = WATCH_INTERVAL,
    ):
        self.db = db
        self.incoming_dir = Path(incoming_dir or DEFAULT_INCOMING_DIR)
        self.poll_interval = poll_interval
        self.is_running = False
        self._last_sizes: Dict[str, Tuple[int, float]] = {}

        # Ensure directory exists
        self.incoming_dir.mkdir(parents=True, exist_ok=True)

    def is_file_ready(self, file_path: Path) -> bool:
        """
        Checks if a file is completely written and not still being copied/downloaded.
        """
        try:
            current_size = file_path.stat().st_size
            now = time.time()
            path_str = str(file_path)

            if path_str in self._last_sizes:
                last_size, last_time = self._last_sizes[path_str]
                # If size hasn't changed in at least 1 second, it is ready
                if current_size == last_size and (now - last_time) >= 1.0:
                    del self._last_sizes[path_str]
                    return True

            self._last_sizes[path_str] = (current_size, now)
            return False
        except (OSError, FileNotFoundError):
            return False

    def scan_once(self) -> List[Dict[str, Any]]:
        """
        Performs a single pass scan of the incoming folder.
        Processes ready files and returns operation results.
        """
        results: List[Dict[str, Any]] = []
        if not self.incoming_dir.exists():
            return results

        entries = sorted(self.incoming_dir.iterdir(), key=lambda f: f.stat().st_mtime if f.exists() else 0)

        for item in entries:
            if not item.is_file():
                continue

            name = item.name
            if name.startswith(".") or name.endswith(".failed"):
                continue

            # Check transient extensions
            if any(name.endswith(ext) for ext in TRANSIENT_EXTENSIONS):
                continue

            # Check supported extensions (handle compound extensions like .tar.gz)
            is_supported = any(
                name.lower().endswith(ext) or item.suffix.lower() == ext
                for ext in SUPPORTED_EXTENSIONS
            )
            if not is_supported:
                continue

            # Check if file has finished writing
            if not self.is_file_ready(item):
                continue

            # File is ready, process it
            logger.info(f"Rilevato file in drop folder: {item.name}")
            try:
                summary = ingest_source(item, self.db, provider="auto")
                # Remove file upon successful processing as required by Issue #4
                item.unlink(missing_ok=True)
                summary["action"] = "processed_and_deleted"
                summary["file_name"] = item.name
                results.append(summary)
                logger.info(
                    f"✓ Completato ingest per '{item.name}': "
                    f"{summary['imported']} importate, {summary['skipped']} saltate."
                )
            except Exception as e:
                logger.error(f"Errore durante l'ingestione di {item.name}: {e}", exc_info=True)
                failed_name = item.with_suffix(item.suffix + ".failed")
                try:
                    item.rename(failed_name)
                    # Write error reason file
                    err_file = failed_name.with_suffix(".err.txt")
                    err_file.write_text(str(e), encoding="utf-8")
                except Exception:
                    pass
                results.append({
                    "status": "error",
                    "file_name": item.name,
                    "error": str(e),
                })

        return results

    def run_forever(self):
        """Continuous polling loop."""
        self.is_running = True
        logger.info(f"Avviato watcher drop folder su: {self.incoming_dir} (intervallo {self.poll_interval}s)")
        try:
            while self.is_running:
                self.scan_once()
                time.sleep(self.poll_interval)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self):
        self.is_running = False
        logger.info("Watcher drop folder arrestato.")
