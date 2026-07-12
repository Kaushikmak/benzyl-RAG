"""CitationAgent responsible for extracting structured citations from grounded context chunks."""

from typing import Any, Dict, List
from app.agents.models import Citation


class CitationAgent:
    """Associates candidate context with supporting sources returning structured Citation models."""

    def extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """Extract structured Citation objects from retrieved chunks."""
        citations: List[Citation] = []
        for chunk in chunks:
            source = str(chunk.get("source", "unknown"))
            snippet = str(chunk.get("content", "")).strip()[:200]
            score = float(chunk.get("rerank_score", chunk.get("score", 1.0)))
            citations.append(Citation(source=source, snippet=snippet, relevance_score=score))
        return citations
