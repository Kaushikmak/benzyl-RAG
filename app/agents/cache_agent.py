"""CacheAgent implementing lightweight cache abstraction with Redis-ready interface."""

from typing import Any, Dict, Optional


class CacheAgent:
    """Cache storage abstraction for queries, retrieval results, and synthesized reports."""

    def __init__(self, backend: Optional[Any] = None):
        self._memory_cache: Dict[str, Any] = {}
        self.backend = backend

    def get(self, key: str) -> Optional[Any]:
        """Get cached item by key."""
        if self.backend and hasattr(self.backend, "get"):
            return self.backend.get(key)
        return self._memory_cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store item in cache."""
        if self.backend and hasattr(self.backend, "set"):
            self.backend.set(key, value)
        else:
            self._memory_cache[key] = value

    def clear(self) -> None:
        """Clear all cached entries."""
        self._memory_cache.clear()
