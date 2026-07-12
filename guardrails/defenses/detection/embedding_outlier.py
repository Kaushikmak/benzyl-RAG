"""Embedding outlier detector flagging anomalous candidates before score normalization."""

import logging
import numpy as np
from typing import Callable, List, Optional
from guardrails.core.base import Candidate, DefenseStage

logger = logging.getLogger(__name__)


class EmbeddingOutlierDetector(DefenseStage):
    """Detects candidates whose embedding deviates strongly from the candidate cluster centroid."""

    def __init__(
        self,
        max_z_score: float = 2.0,
        embed_func: Optional[Callable[[str], List[float]]] = None,
    ):
        self.max_z_score = max_z_score
        self.embed_func = embed_func

    def filter_candidates(
        self,
        candidates: List[Candidate],
        query: str,
        embeddings: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> List[Candidate]:
        """Flag or filter candidates whose distance to the cluster centroid is an outlier."""
        if len(candidates) < 3:
            return candidates

        embs = []
        if embeddings and len(embeddings) == len(candidates):
            embs = [np.array(e, dtype=np.float32) for e in embeddings]
        elif self.embed_func:
            embs = [np.array(self.embed_func(c.content), dtype=np.float32) for c in candidates]

        if not embs:
            return candidates

        matrix = np.stack(embs)
        centroid = np.mean(matrix, axis=0)
        dists = np.linalg.norm(matrix - centroid, axis=1)

        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        if std_dist < 1e-6:
            return candidates

        admitted = []
        for cand, dist in zip(candidates, dists):
            z_score = float((dist - mean_dist) / std_dist)
            cand.scores["outlier_z"] = z_score
            if z_score <= self.max_z_score:
                admitted.append(cand)
            else:
                cand.quarantined = True
                cand.quarantine_reason = f"Embedding outlier (z={z_score:.2f} > {self.max_z_score})"
                logger.info(f"Quarantined candidate '{cand.doc_id}': {cand.quarantine_reason}")

        return admitted
