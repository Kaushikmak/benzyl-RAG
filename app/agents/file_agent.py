"""Specialized FileAgent for handling filesystem operations, markdown export, and atomic file writes."""

import logging
import os
import re
from typing import Any, Dict, Optional
from app.agents.utils import atomic_write_file

logger = logging.getLogger(__name__)


class FileAgent:
    """Parses natural language file action intents and safely executes them with atomic writes."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)

    def validate_target_path(
        self, filename: str, whitelist_root: Optional[str] = None
    ) -> str:
        """Validate that filename resolves strictly under whitelist_root (following symlinks).
        Raises SecurityException if path traversal or symlink escape outside whitelist root is detected.
        """
        from pathlib import Path
        from app.agents.exceptions import SecurityException

        root_str = whitelist_root or self.workspace_root
        resolved_root = Path(root_str).resolve()

        target = Path(filename)
        if target.is_absolute():
            cand = target
        else:
            cand = resolved_root / target

        if cand.exists():
            resolved_target = cand.resolve()
        else:
            resolved_target = cand.parent.resolve() / cand.name

        try:
            resolved_target.relative_to(resolved_root)
        except ValueError:
            raise SecurityException(
                f"Security Error: Target path '{filename}' ({resolved_target}) resolves outside whitelist root '{resolved_root}'."
            )
        return str(resolved_target)

    def _sanitize_path(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal outside workspace root."""
        return self.validate_target_path(filename, whitelist_root=self.workspace_root)

    def parse_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse natural language query for file action instructions."""
        q = query.strip()

        delete_match = re.search(
            r"\b(delete|remove|rm)\s+(the\s+)?(file\s+)?([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]+)\b",
            q,
            re.IGNORECASE,
        )
        if delete_match:
            target_file = delete_match.group(4)
            return {
                "action": "DELETE",
                "target_file": target_file,
                "core_query": "",
            }

        save_match = re.search(
            r"(.*?)(?:and\s+)?\b(save|write|export|output|dump|create)\b.*?\b(?:file\s+(?:named|as)\s+|name\s+(?:that\s+)?file\s+as\s+|to\s+file\s+named\s+|to\s+file\s+|to\s+|in\s+)([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]{2,4})\b.*$",
            q,
            re.IGNORECASE | re.DOTALL,
        )
        if save_match:
            core_query = save_match.group(1).strip()
            core_query = re.sub(r"\s+and$", "", core_query, flags=re.IGNORECASE).strip()
            target_file = save_match.group(3).strip()
            return {
                "action": "SAVE",
                "target_file": target_file,
                "core_query": core_query if core_query else q,
            }

        simple_save = re.search(
            r"^(save|write|export|create)\s+(.+?)\s+(?:to|in|as)\s+([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]{2,4})$",
            q,
            re.IGNORECASE,
        )
        if simple_save:
            core_query = simple_save.group(2).strip()
            target_file = simple_save.group(3).strip()
            return {
                "action": "SAVE",
                "target_file": target_file,
                "core_query": core_query,
            }

        general_save = re.search(
            r"^(.*?)\s*\b(save|write|export|output|dump|create|put)\b.*?\b(?:in|to|as|named)?\s*(?:file\s+)?([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]{2,4})\b(.*)$",
            q,
            re.IGNORECASE | re.DOTALL,
        )
        if general_save:
            pre_text = general_save.group(1).strip()
            post_text = general_save.group(4).strip()
            target_file = general_save.group(3).strip()
            core_query = f"{pre_text} {post_text}".strip()
            core_query = re.sub(r"\bfile\b", "", core_query, flags=re.IGNORECASE).strip()
            return {
                "action": "SAVE",
                "target_file": target_file,
                "core_query": core_query if core_query else q,
            }

        return None

    def export_markdown(self, filename: str, content: str) -> str:
        """Export Markdown content to disk atomically."""
        return self.execute("SAVE", filename, content=content)

    def execute(
        self, action: str, target_file: str, content: Optional[str] = None
    ) -> str:
        """Safely execute the requested filesystem operation using atomic writes (.tmp -> flush -> fsync -> os.replace)."""
        abs_path = self._sanitize_path(target_file)
        rel_path = os.path.relpath(abs_path, self.workspace_root)

        action_upper = action.upper()
        if action_upper in ("DELETE", "REMOVE"):
            if os.path.exists(abs_path):
                try:
                    os.remove(abs_path)
                    logger.info("FileAgent deleted file: %s", rel_path)
                    return f"[FileAgent] Successfully deleted file: {rel_path}"
                except Exception as e:
                    return f"[FileAgent] Error deleting file {rel_path}: {e}"
            else:
                return f"[FileAgent] File not found for deletion: {rel_path}"

        elif action_upper in ("SAVE", "CREATE", "WRITE", "EXPORT"):
            try:
                data = content or ""
                atomic_write_file(abs_path, data, workspace_root=self.workspace_root)
                byte_len = len(data.encode("utf-8"))
                logger.info("FileAgent saved file: %s (%d bytes)", rel_path, byte_len)
                return (
                    f"[FileAgent] Successfully saved file: {rel_path} ({byte_len} bytes written)"
                )
            except Exception as e:
                return f"[FileAgent] Error saving file {rel_path}: {e}"

        elif action_upper == "APPEND":
            try:
                os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
                data = content or ""
                with open(abs_path, "a", encoding="utf-8") as f:
                    f.write(data)
                byte_len = len(data.encode("utf-8"))
                logger.info("FileAgent appended file: %s (%d bytes)", rel_path, byte_len)
                return (
                    f"[FileAgent] Successfully appended to file: {rel_path} ({byte_len} bytes written)"
                )
            except Exception as e:
                return f"[FileAgent] Error appending to file {rel_path}: {e}"

        return f"[FileAgent] Unknown file action: {action}"
