"""Partition-Aggregate Isolation Mode (RobustRAG-style defense against single-source poisoning)."""

import logging
from typing import Callable, List
from guardrails.core.base import Candidate

logger = logging.getLogger(__name__)


class PartitionAggregateIsolation:
    """Partitions candidates into independent sets, generates per-partition sub-answers, and aggregates."""

    def __init__(self, num_partitions: int = 2):
        self.num_partitions = max(2, num_partitions)

    def partition_candidates(self, candidates: List[Candidate]) -> List[List[Candidate]]:
        """Partition candidate list into non-overlapping groups."""
        if not candidates:
            return []

        partitions: List[List[Candidate]] = [[] for _ in range(min(self.num_partitions, len(candidates)))]
        for idx, cand in enumerate(candidates):
            partitions[idx % len(partitions)].append(cand)
        return partitions

    def generate_isolated_answer(
        self,
        query: str,
        candidates: List[Candidate],
        sub_generate_fn: Callable[[str, List[Candidate]], str],
        aggregate_fn: Callable[[str, List[str]], str],
    ) -> str:
        """Run partition -> generate -> aggregate pipeline."""
        partitions = self.partition_candidates(candidates)
        sub_answers: List[str] = []

        for p_idx, part in enumerate(partitions):
            logger.info(f"Generating sub-answer for partition #{p_idx+1}/{len(partitions)} ({len(part)} chunks)")
            ans = sub_generate_fn(query, part)
            sub_answers.append(ans)

        logger.info("Aggregating sub-answers across partitions...")
        final_answer = aggregate_fn(query, sub_answers)
        return final_answer
