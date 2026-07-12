"""RerankerAgent responsible for cross-encoder reranking of candidate chunks."""

from typing import Any, Callable, Dict, List, Optional


class RerankerAgent:
    """Reranks candidate chunks based on query relevance supporting configurable top-k."""

    def __init__(
        self,
        reranker_callback: Optional[Callable[[str, List[Dict[str, Any]], int], List[Dict[str, Any]]]] = None,
    ):
        self.reranker_callback = reranker_callback

    def rerank(
        self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank chunks and return top_k candidates."""
        if self.reranker_callback:
            return self.reranker_callback(query, chunks, top_k)

        # Sort by existing score descending if no callback provided
        sorted_chunks = sorted(
            chunks,
            key=lambda x: float(x.get("rerank_score", x.get("score", 0.0))),
            reverse=True,
        )
        return sorted_chunks[:top_k]
