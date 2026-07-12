"""ObservabilityAgent responsible for collecting execution telemetry and mission metrics."""

from typing import Dict
from app.agents.models import MissionMetrics


class ObservabilityAgent:
    """Collects and returns structured MissionMetrics across pipeline stages."""

    def record_metrics(
        self,
        timestamps: Dict[str, float],
        chunk_count: int = 0,
        cache_hit: bool = False,
        security_violations: int = 0,
        approval_requests: int = 0,
        failures: int = 0,
    ) -> MissionMetrics:
        """Compute latency intervals from stage timestamps and return structured MissionMetrics."""
        t_start = timestamps.get("start", 0.0)
        t_retrieved = timestamps.get("retrieved", t_start)
        t_reranked = timestamps.get("reranked", t_retrieved)
        t_synthesized = timestamps.get("synthesized", t_reranked)
        t_end = timestamps.get("end", t_synthesized)

        retrieval_ms = round((t_retrieved - t_start) * 1000, 2)
        reranking_ms = round((t_reranked - t_retrieved) * 1000, 2)
        synthesis_ms = round((t_synthesized - t_reranked) * 1000, 2)
        total_ms = round((t_end - t_start) * 1000, 2)

        return MissionMetrics(
            retrieval_ms=max(0.0, retrieval_ms),
            reranking_ms=max(0.0, reranking_ms),
            synthesis_ms=max(0.0, synthesis_ms),
            total_ms=max(0.0, total_ms),
            chunk_count=chunk_count,
            cache_hit=cache_hit,
            security_violations=security_violations,
            approval_requests=approval_requests,
            failures=failures,
        )
