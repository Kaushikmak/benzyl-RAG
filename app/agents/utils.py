"""Utility functions for persistence, atomic file I/O, and snapshot management."""

import json
import logging
import os
from typing import Optional
from app.agents.exceptions import MissionNotFound
from app.agents.models import MissionSnapshot

logger = logging.getLogger(__name__)


def atomic_write_file(target_path: str, content: str, workspace_root: Optional[str] = None) -> str:
    """Write file atomically using .tmp -> flush -> fsync -> os.replace."""
    if os.path.isabs(target_path):
        abs_target = os.path.abspath(target_path)
    else:
        root = os.path.abspath(workspace_root or ".")
        abs_target = os.path.abspath(os.path.join(root, target_path))

    parent_dir = os.path.dirname(abs_target) or "."
    os.makedirs(parent_dir, exist_ok=True)

    tmp_path = f"{abs_target}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, abs_target)
        logger.info("Atomic write complete: %s (%d bytes)", abs_target, len(content.encode("utf-8")))
        return abs_target
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise e


def _get_snapshot_path(request_id: str, base_dir: str = ".mission_state") -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{request_id}.json")


def save_snapshot(snapshot: MissionSnapshot, base_dir: str = ".mission_state") -> str:
    """Save mission snapshot to disk formatted as JSON."""
    path = _get_snapshot_path(snapshot.request_id, base_dir=base_dir)
    data = snapshot.model_dump(mode="json")
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    atomic_write_file(path, json_str)
    logger.info("Saved mission snapshot to %s", path)
    return path


def load_snapshot(request_id: str, base_dir: str = ".mission_state") -> MissionSnapshot:
    """Load mission snapshot from disk by request_id."""
    path = _get_snapshot_path(request_id, base_dir=base_dir)
    if not os.path.exists(path):
        raise MissionNotFound(f"Mission snapshot not found for request_id: {request_id}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return MissionSnapshot.model_validate(data)


def update_snapshot(snapshot: MissionSnapshot, base_dir: str = ".mission_state") -> str:
    """Update existing mission snapshot on disk."""
    return save_snapshot(snapshot, base_dir=base_dir)
