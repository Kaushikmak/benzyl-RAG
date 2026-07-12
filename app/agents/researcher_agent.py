"""ResearcherAgent responsible solely for document retrieval (no synthesis or ranking)."""

from typing import Any, Callable, Dict, List, Optional


class ResearcherAgent:
    """Retrieves chunks from local knowledge base or callback retriever without ranking or synthesis."""

    def __init__(self, retriever_callback: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None):
        self.retriever_callback = retriever_callback

    def retrieve(self, query: str, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve candidate chunks for the query according to the retrieval plan."""
        if self.retriever_callback:
            return self.retriever_callback(query, plan)

        # Mock retrieval when callback is not provided
        return [
            {
                "id": "mock_chunk_1",
                "content": f"The Google File System (GFS) architecture uses a large 64 MB chunk size to optimize for multi-GB files regarding: {query}",
                "source": "gfs.pdf",
                "score": 0.85,
            },
            {
                "id": "mock_chunk_2",
                "content": "GFS assumes component failures are the norm and implements automatic self-recovery across inexpensive commodity storage servers.",
                "source": "gfs.pdf",
                "score": 0.78,
            },
        ]
