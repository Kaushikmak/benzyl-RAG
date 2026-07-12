"""SynthesisAgent responsible for generating grounded Markdown response from retrieved context."""

from typing import Any, Callable, Dict, List, Optional
from app.agents.models import Citation


class SynthesisAgent:
    """Synthesizes markdown response grounded strictly in verified context and citations."""

    def __init__(
        self,
        llm_callback: Optional[
            Callable[[str, List[Dict[str, Any]], List[Citation]], str]
        ] = None,
    ):
        self.llm_callback = llm_callback

    def synthesize(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        citations: List[Citation],
    ) -> str:
        """Generate grounded Markdown synthesis."""
        if not chunks:
            return f"No grounded evidence found in the document repository for query: '{query}'."

        if self.llm_callback:
            return self.llm_callback(query, chunks, citations)

        lines = [
            f"# Synthesis Report: *{query}*\n",
            "### Grounded Findings\n",
        ]

        for i, chunk in enumerate(chunks[:5], 1):
            src = chunk.get("source", "unknown")
            text = str(chunk.get("content", "")).strip()
            lines.append(f"{i}. **[{src}]**: {text}\n")

        if citations:
            lines.append("### Verified Citations\n")
            for c in citations[:5]:
                lines.append(f"- `{c.source}` *(Relevance: {c.relevance_score:.2f})*")

        return "\n".join(lines)
