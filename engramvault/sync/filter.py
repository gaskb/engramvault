import fnmatch
import logging
import os
import urllib.parse
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger("engramvault.sync.filter")

# Sentinel filenames that mark a directory as confidential / never sync
SENTINEL_IGNORE_FILENAMES = {
    ".no-vault",
    ".no-engramvault",
    ".engramvaultignore",
    ".memoriaignore",
}

# Standard paths to search for user-configured ignore patterns
DEFAULT_IGNORE_FILES = [
    Path.home() / ".engramvault" / "sync_ignore.txt",
    Path.home() / ".memoria" / "sync_ignore.txt",
    Path.home() / ".engramvaultignore",
]


class SyncFilter:
    """
    Evaluates whether a session/workspace should be excluded from synchronization.
    Supports:
    1. Sentinel files in project folder (.no-vault, .engramvaultignore, .no-engramvault).
    2. Ignore rule patterns loaded from file (~/.engramvault/sync_ignore.txt).
    3. Custom patterns passed via CLI / environment variable (ENGRAMVAULT_SYNC_EXCLUDE).
    """

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        ignore_file: Optional[Path | str] = None,
    ):
        self.patterns: List[str] = []
        self._checked_dirs_cache: dict[str, bool] = {}

        # 1. Load patterns from explicit or default ignore file
        if ignore_file:
            self.load_ignore_file(Path(ignore_file))
        else:
            for cand in DEFAULT_IGNORE_FILES:
                if cand.is_file():
                    self.load_ignore_file(cand)
                    break

        # 2. Add patterns from env var
        env_excludes = os.environ.get("ENGRAMVAULT_SYNC_EXCLUDE", os.environ.get("MEMORIA_SYNC_EXCLUDE", ""))
        if env_excludes:
            for item in env_excludes.split(","):
                clean = item.strip()
                if clean and clean not in self.patterns:
                    self.patterns.append(clean)

        # 3. Add explicit programmatic patterns
        if patterns:
            for pat in patterns:
                clean = pat.strip()
                if clean and clean not in self.patterns:
                    self.patterns.append(clean)

    def load_ignore_file(self, file_path: Path):
        """Loads glob patterns from a newline-separated file (supports # comments)."""
        if not file_path.is_file():
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line not in self.patterns:
                        self.patterns.append(line)
        except Exception as e:
            logger.warning(f"Errore lettura file ignore {file_path}: {e}")

    def has_sentinel_ignore_file(self, folder_path: Path) -> bool:
        """Checks if folder or any of its parent directories contains a sentinel ignore file."""
        if not folder_path or not folder_path.exists():
            return False

        path_key = str(folder_path.resolve())
        if path_key in self._checked_dirs_cache:
            return self._checked_dirs_cache[path_key]

        curr = folder_path.resolve() if folder_path.is_dir() else folder_path.resolve().parent
        # Walk up to root, checking each level for sentinel files
        root = Path(curr.anchor)
        is_ignored = False
        while curr != root:
            for sentinel in SENTINEL_IGNORE_FILENAMES:
                if (curr / sentinel).exists():
                    is_ignored = True
                    break
            if is_ignored:
                break
            if curr.parent == curr:
                break
            curr = curr.parent

        self._checked_dirs_cache[path_key] = is_ignored
        return is_ignored

    def matches_pattern(self, text: str) -> bool:
        """Matches a string (path, name, uri) against registered glob patterns (case-insensitive)."""
        if not text:
            return False
        clean = text.strip()
        lower_clean = clean.lower()
        for pat in self.patterns:
            pat_lower = pat.lower()
            # Normalize leading/trailing globs
            if fnmatch.fnmatch(lower_clean, pat_lower) or fnmatch.fnmatch(lower_clean, f"*{pat_lower}*"):
                return True
        return False

    def is_workspace_ignored(self, workspace_path: Optional[str | Path]) -> bool:
        """
        Determines whether a workspace directory or URI is excluded.
        Handles file:// URIs, absolute paths, and pattern matching.
        """
        if not workspace_path:
            return False

        str_path = str(workspace_path).strip()

        # Handle file:// URI
        if str_path.startswith("file://"):
            parsed = urllib.parse.urlparse(str_path)
            str_path = urllib.parse.unquote(parsed.path)

        # Check pattern match against the path string
        if self.matches_pattern(str_path):
            return True

        # Check file system sentinel file
        p = Path(str_path)
        if p.exists() and self.has_sentinel_ignore_file(p):
            return True

        return False
