"""Detection filters (Graph gate, embedding outliers, near-duplicate clustering)."""

from guardrails.defenses.detection.graph_gate import GraphExpansionGate, filter_neighbors_by_query_relevance
from guardrails.defenses.detection.embedding_outlier import EmbeddingOutlierDetector
from guardrails.defenses.detection.near_dup_cluster import NearDuplicateClusterDetector

__all__ = [
    "GraphExpansionGate",
    "filter_neighbors_by_query_relevance",
    "EmbeddingOutlierDetector",
    "NearDuplicateClusterDetector",
]
