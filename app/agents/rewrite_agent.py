"""RewriteAgent responsible for optimizing user queries for retrieval."""

import re
from typing import Any, Dict


class RewriteAgent:
    """Rewrites vague or conversational user queries into dense/sparse retrieval-optimized queries."""

    def rewrite(self, query: str, plan: Dict[str, Any]) -> str:
        """Optimize user query for vector and BM25 search."""
        cleaned = query.strip()
        # Remove conversational filler prefix
        cleaned = re.sub(
            r"^(please\s+|can\s+you\s+|could\s+you\s+|tell\s+me\s+about\s+|what\s+is\s+|how\s+does\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            return query.strip()
        return cleaned
