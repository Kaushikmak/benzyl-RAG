"""
Atomic lockfile implementation for single-writer index protection.

Rules:
  - Acquire uses os.O_CREAT | os.O_EXCL for atomic creation (no check-then-create race).
  - Lock file stores JSON: {"pid": <int>, "timestamp": <float>}.
  - On acquisition failure, reads the lock's PID and timestamp:
      - If PID is not running (kill(pid, 0) raises ProcessLookupError) -> stale -> auto-clear.
      - If timestamp is older than staleness_seconds -> stale -> auto-clear and log warning.
      - Otherwise -> live lock -> raise LockAlreadyHeldError with lock age and PID.
  - Release is done in a finally-guarded block; atexit is registered as backstop.
  - Implements the context-manager protocol (__enter__ / __exit__).
"""

import atexit
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LockAlreadyHeldError(Exception):
    """Raised when a live (non-stale) lock is held by another process."""


class IndexLockfile:
    """Atomic file-based write lock protecting single-writer index operations."""

    def __init__(self, lock_path: str, staleness_seconds: float = 1800.0):
        """
        Args:
            lock_path: Path to the lock file (e.g. .data/index.lock).
            staleness_seconds: Age (seconds) after which a lock is considered stale
                               even if the PID appears to still be running.
                               Default: 1800 (30 minutes).
        """
        self.lock_path = lock_path
        self.staleness_seconds = staleness_seconds
        self._held = False
        atexit.register(self._atexit_release)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_boot_id(self) -> Optional[str]:
        """Return system boot ID if readable, or None."""
        try:
            with open("/proc/sys/kernel/random/boot_id", "r") as f:
                return f.read().strip()
        except Exception:
            return None

    def _get_container_id(self) -> Optional[str]:
        """Return container identifier or hostname if in Docker/container environment."""
        if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
            try:
                import socket
                return socket.gethostname()
            except Exception:
                return "docker-container"
        try:
            with open("/proc/self/cgroup", "r") as f:
                content = f.read()
                if "docker" in content or "containerd" in content or "kubepods" in content:
                    import socket
                    return socket.gethostname()
        except Exception:
            pass
        return None

    def _pid_is_running(self, pid: int) -> bool:
        """Return True if process with given PID exists."""
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            # ProcessLookupError -> PID not found (dead)
            # PermissionError   -> PID exists but we can't signal it (still running)
            return not isinstance(BaseException(), ProcessLookupError)
        except Exception:
            return False

    def _read_lock(self):
        """Return (pid, timestamp, boot_id, container_id) from lock file, or (None, None, None, None) on parse failure."""
        try:
            with open(self.lock_path, "r") as f:
                data = json.load(f)
            return (
                int(data.get("pid", 0)),
                float(data.get("timestamp", 0.0)),
                data.get("boot_id"),
                data.get("container_id"),
            )
        except Exception:
            return None, None, None, None

    def _write_lock(self) -> None:
        """Atomically create lock file with current PID, timestamp, boot_id, and container_id."""
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        fd = os.open(
            self.lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        )
        try:
            payload = json.dumps({
                "pid": os.getpid(),
                "timestamp": time.time(),
                "boot_id": self._get_boot_id(),
                "container_id": self._get_container_id(),
            }).encode()
            os.write(fd, payload)
        finally:
            os.close(fd)

    def _try_clear_stale(self) -> bool:
        """Inspect current lock file; clear it if stale.  Return True if cleared."""
        pid, ts, lock_boot_id, lock_container_id = self._read_lock()
        if pid is None:
            # Can't parse lock -> treat as stale
            try:
                os.remove(self.lock_path)
                logger.warning("IndexLockfile: removed unparseable stale lock '%s'.", self.lock_path)
            except FileNotFoundError:
                pass
            return True

        curr_boot_id = self._get_boot_id()
        if lock_boot_id and curr_boot_id and lock_boot_id != curr_boot_id:
            logger.warning(
                "IndexLockfile: lock held across system reboot (boot_id mismatch). Auto-clearing."
            )
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            return True

        curr_container_id = self._get_container_id()
        if lock_container_id and curr_container_id and lock_container_id != curr_container_id:
            logger.warning(
                "IndexLockfile: lock held by previous container instance ('%s' vs current '%s'). Auto-clearing.",
                lock_container_id, curr_container_id,
            )
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            return True

        age = time.time() - (ts or 0.0)
        pid_running = self._pid_is_running(pid)

        if not pid_running:
            logger.warning(
                "IndexLockfile: lock held by dead PID %d (age %.0fs). Auto-clearing.",
                pid, age,
            )
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            return True

        if age > self.staleness_seconds:
            logger.warning(
                "IndexLockfile: lock held by PID %d is %.0fs old (threshold %.0fs). Auto-clearing as stale.",
                pid, age, self.staleness_seconds,
            )
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
            return True

        # Live lock
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Acquire the index write lock.

        Raises:
            LockAlreadyHeldError: If a live (non-stale) lock is currently held.
        """
        try:
            self._write_lock()
            self._held = True
            logger.debug("IndexLockfile: acquired '%s' (PID %d).", self.lock_path, os.getpid())
        except FileExistsError:
            # Lock file already exists -> inspect staleness
            cleared = self._try_clear_stale()
            if cleared:
                # Retry once after clearing
                try:
                    self._write_lock()
                    self._held = True
                    logger.debug("IndexLockfile: acquired '%s' after clearing stale lock.", self.lock_path)
                    return
                except FileExistsError:
                    pass  # fall through to raise

            pid, ts, boot_id, container_id = self._read_lock()
            age = time.time() - (ts or time.time())
            raise LockAlreadyHeldError(
                f"Index lock is held by PID {pid} (age {age:.0f}s). "
                "Wait for it to finish, or remove the lock manually if the process crashed."
            )

    def release(self) -> None:
        """Release the index write lock (safe to call even if not held)."""
        if self._held:
            try:
                os.remove(self.lock_path)
                logger.debug("IndexLockfile: released '%s'.", self.lock_path)
            except FileNotFoundError:
                pass
            finally:
                self._held = False

    def is_locked(self) -> bool:
        """Return True if a non-stale lock exists (useful for query-time checks)."""
        if not os.path.exists(self.lock_path):
            return False
        pid, ts, boot_id, container_id = self._read_lock()
        if pid is None:
            return False
        curr_boot = self._get_boot_id()
        if boot_id and curr_boot and boot_id != curr_boot:
            return False
        curr_container = self._get_container_id()
        if container_id and curr_container and container_id != curr_container:
            return False
        age = time.time() - (ts or 0.0)
        if age > self.staleness_seconds:
            return False
        return self._pid_is_running(pid)

    def warn_if_locked(self) -> None:
        """Log a clear warning if an index write is currently in progress."""
        if self.is_locked():
            pid, ts, boot_id, container_id = self._read_lock()
            age = time.time() - (ts or time.time())
            logger.warning(
                "[benzene-rag] An index write is currently in progress (PID %d, age %.0fs). "
                "Query results may reflect a partially updated index until indexing completes.",
                pid, age,
            )

    def _atexit_release(self) -> None:
        """Backstop: release lock on interpreter shutdown if still held."""
        self.release()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "IndexLockfile":
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        self.release()
