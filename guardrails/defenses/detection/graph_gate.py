"""Graph-Expansion Gate: Filters graph neighbor chunks by query similarity floor threshold."""

import logging
import numpy as np
from typing import Callable, Dict, List, Optional
from guardrails.core.base import Candidate, DefenseStage

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_GATE_THRESHOLD = 0.25


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


class GraphExpansionGate(DefenseStage):
    """Gated Graph Expansion filter preventing poisoned adjacent links from entering Stage 3.4."""

    def __init__(
        self,
        threshold: float = DEFAULT_GRAPH_GATE_THRESHOLD,
        embed_func: Optional[Callable[[str], List[float]]] = None,
    ):
        self.threshold = threshold
        self.embed_func = embed_func

    def filter_candidates(
        self,
        candidates: List[Candidate],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Candidate]:
        """Filter out candidates whose similarity to query is below `self.threshold`."""
        if not candidates:
            return []

        if query_embedding is None and self.embed_func:
            query_embedding = np.array(self.embed_func(query), dtype=np.float32)

        admitted = []
        for cand in candidates:
            # If candidate already has a precomputed vector score or similarity
            sim = cand.scores.get("query_sim", cand.scores.get("vector", None))

            if sim is None and query_embedding is not None and self.embed_func:
                cand_emb = np.array(self.embed_func(cand.content), dtype=np.float32)
                sim = _cosine_similarity(query_embedding, cand_emb)
                cand.scores["query_sim"] = sim

            if sim is None:
                # Fallback if no embedding function or score provided
                sim = 1.0 if query.lower() in cand.content.lower() else 0.15

            if sim >= self.threshold:
                cand.scores["graph_gate_sim"] = sim
                admitted.append(cand)
            else:
                logger.info(
                    f"Graph gate rejected candidate '{cand.doc_id}' "
                    f"(sim={sim:.4f} < {self.threshold})"
                )

        return admitted

    def sweep_thresholds(
        self,
        candidates: List[Candidate],
        query: str,
        thresholds: List[float],
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[float, Dict[str, float]]:
        """Sweep similarity thresholds to evaluate precision/recall tradeoffs."""
        results = {}
        for t in thresholds:
            original_t = self.threshold
            self.threshold = t
            filtered = self.filter_candidates(candidates, query, query_embedding)
            self.threshold = original_t

            admit_ratio = len(filtered) / len(candidates) if candidates else 0.0
            results[t] = {
                "admitted_count": len(filtered),
                "total_count": len(candidates),
                "admit_ratio": round(admit_ratio, 4),
            }
        return results


def filter_neighbors_by_query_relevance(
    neighbors: List[Candidate],
    query: str,
    threshold: float = DEFAULT_GRAPH_GATE_THRESHOLD,
    embed_func: Optional[Callable[[str], List[float]]] = None,
) -> List[Candidate]:
    gate = GraphExpansionGate(threshold=threshold, embed_func=embed_func)
    return gate.filter_candidates(neighbors, query=query)
