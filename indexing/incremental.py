"""
Incremental indexing manifest ledger.

Stores per-source-file: relative path, mtime, SHA-256 content hash, and the set
of chunk / vector IDs produced from that file. On each index run:
  - Files whose mtime AND hash are unchanged are skipped (no re-embed).
  - Files whose mtime or hash differ are treated as modify = delete-then-reinsert.
  - Files present in the manifest but absent from the data directory are treated as deletions.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1


class IndexManifest:
    """Persistent JSON manifest tracking per-file mtime, SHA-256 hash, and chunk IDs."""

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self._data: Dict[str, Dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = raw.get("files", {})
            except Exception as exc:
                logger.warning("IndexManifest: failed to load manifest (%s) – starting fresh.", exc)
                self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.manifest_path) or ".", exist_ok=True)
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": _MANIFEST_VERSION, "files": self._data}, f, indent=2)
        os.replace(tmp, self.manifest_path)

    @staticmethod
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _mtime(path: str) -> float:
        return os.stat(path).st_mtime

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_unchanged(self, path: str) -> bool:
        """Return True iff path is tracked and both mtime AND hash match current disk state."""
        if not os.path.exists(path):
            return False
        entry = self._data.get(path)
        if entry is None:
            return False
        try:
            if entry.get("mtime") != self._mtime(path):
                return False
            if entry.get("sha256") != self._sha256(path):
                return False
            return True
        except Exception:
            return False

    def record(self, path: str, chunk_ids: List[str]) -> None:
        """Record a file as successfully indexed with the given chunk IDs."""
        try:
            self._data[path] = {
                "mtime": self._mtime(path),
                "sha256": self._sha256(path),
                "chunk_ids": list(chunk_ids),
            }
            self._save()
        except Exception as exc:
            logger.error("IndexManifest: failed to record %s – %s", path, exc)

    def remove(self, path: str) -> None:
        """Remove a file entry from the manifest (called after its vectors are purged)."""
        if path in self._data:
            del self._data[path]
            self._save()

    def get_chunk_ids(self, path: str) -> List[str]:
        """Return the chunk IDs previously indexed for path, or []."""
        return list(self._data.get(path, {}).get("chunk_ids", []))

    def find_deleted_entries(self, current_paths: List[str]) -> List[str]:
        """Return manifested paths that are no longer in current_paths (i.e., deleted files)."""
        current_set = set(current_paths)
        return [p for p in self._data if p not in current_set]

    def all_paths(self) -> List[str]:
        """Return all tracked source paths."""
        return list(self._data.keys())
