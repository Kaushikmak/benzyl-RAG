"""CompressionAgent responsible for deduplicating and merging retrieved chunks."""

from typing import Any, Dict, List, Set


class CompressionAgent:
    """Removes redundant chunks and merges similar context snippets."""

    def compress(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate chunks based on source and normalized text content."""
        seen_contents: Set[str] = set()
        compressed: List[Dict[str, Any]] = []

        for chunk in chunks:
            content = str(chunk.get("content", "")).strip()
            norm_content = " ".join(content.lower().split())
            if not norm_content or norm_content in seen_contents:
                continue
            seen_contents.add(norm_content)
            compressed.append(chunk)

        return compressed
