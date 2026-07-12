"""Specialized StatusAgent for system telemetry and index metrics."""

import re
from typing import Any, Dict, Optional


class StatusAgent:
    """Specialized StatusAgent handling system telemetry and document counts."""

    def __init__(self):
        self._status_patterns = [
            r"how\s+many\s+(documents|chunks)",
            r"(index|data|system|guardrails)\s+status",
            r"status\s+of\s+(the\s+)?(system|index|data|rag)",
            r"indexed\s+documents",
        ]

    def try_status_route(
        self, query: str, system_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        q_lower = query.strip().lower().rstrip("?")
        for pattern in self._status_patterns:
            if re.search(pattern, q_lower):
                ctx = system_context or {}
                notes_cnt = ctx.get("notes_count", "N/A")
                chunks_cnt = ctx.get("chunks_count", "N/A")
                guardrails = ctx.get("guardrails_active", True)
                return (
                    "### benzyl-RAG System Telemetry\n\n"
                    f"- **Indexed Documents**: `{notes_cnt}`\n"
                    f"- **Total Chunks**: `{chunks_cnt}`\n"
                    f"- **Guardrails Active**: `{guardrails}`\n"
                    f"- **Retrieval Pipeline**: `4-Stage Multi-Step Funnel (Qdrant + BM25 + Graph + Cross-Encoder)`"
                )
        return None
