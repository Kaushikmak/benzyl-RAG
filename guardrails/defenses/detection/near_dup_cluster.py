"""Near-duplicate cluster detector preventing candidate pool flooding."""

import logging
from typing import List
from guardrails.core.base import Candidate, DefenseStage

logger = logging.getLogger(__name__)


def _jaccard_similarity(text1: str, text2: str) -> float:
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    return float(len(t1.intersection(t2))) / float(len(t1.union(t2)))


class NearDuplicateClusterDetector(DefenseStage):
    """Clusters near-duplicate candidates before RRF and collapses/flags flooding attacks."""

    def __init__(self, similarity_threshold: float = 0.82, max_duplicates_allowed: int = 1):
        self.similarity_threshold = similarity_threshold
        self.max_duplicates_allowed = max_duplicates_allowed

    def filter_candidates(
        self,
        candidates: List[Candidate],
        query: str,
        **kwargs
    ) -> List[Candidate]:
        """Collapse near-identical candidate clusters to prevent RRF flooding."""
        if not candidates:
            return []

        admitted: List[Candidate] = []
        for cand in candidates:
            duplicate_count = 0
            for existing in admitted:
                sim = _jaccard_similarity(cand.content, existing.content)
                if sim >= self.similarity_threshold:
                    duplicate_count += 1

            if duplicate_count < self.max_duplicates_allowed:
                admitted.append(cand)
            else:
                cand.quarantined = True
                cand.quarantine_reason = f"Near-duplicate flooding detected (sim >= {self.similarity_threshold})"
                logger.info(f"Quarantined near-duplicate candidate '{cand.doc_id}'")

        return admitted
