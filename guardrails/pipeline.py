"""Production Guardrails pipeline wrapping RAG retrieval with detection and security filtering."""

import logging
from typing import Dict, List
from guardrails.core.base import Candidate, BaseRetriever
from guardrails.defenses.detection import GraphExpansionGate, EmbeddingOutlierDetector, NearDuplicateClusterDetector
from guardrails.defenses.filtering import PromptSanitizer, build_secure_context, build_secure_system_prompt

logger = logging.getLogger(__name__)


class GuardrailsPipeline(BaseRetriever):
    """Wraps RAG retrieval with multi-layer guardrail defenses (deduplication, outlier quarantine, graph gate, sanitization)."""

    def __init__(self, rag_instance=None):
        if rag_instance is None:
            from app.rag import ImprovedRAG
            self.rag = ImprovedRAG()
        else:
            self.rag = rag_instance

        self.graph_gate = GraphExpansionGate(threshold=0.25)
        self.outlier_detector = EmbeddingOutlierDetector(max_z_score=2.0)
        self.neardup_detector = NearDuplicateClusterDetector(similarity_threshold=0.82)
        self.sanitizer = PromptSanitizer()

    def _doc_to_candidate(self, doc) -> Candidate:
        cand = Candidate(
            doc_id=doc.doc_id,
            content=doc.content,
            source=doc.source,
            scores={
                "hybrid": getattr(doc, "hybrid_score", 0.0),
                "vector": getattr(doc, "vector_score", 0.0),
                "bm25": getattr(doc, "bm25_score", 0.0),
                "rerank": getattr(doc, "rerank_score", 0.0),
            },
            metadata=getattr(doc, "metadata", {}),
        )
        return cand

    def retrieve(
        self,
        query: str,
        k: int = 10,
        enable_graph_gate: bool = True,
        enable_outlier: bool = True,
        enable_neardup: bool = True,
        enable_sanitizer: bool = True,
        **kwargs
    ) -> List[Candidate]:
        """Retrieve candidates and pass them through configured Guardrails defense layers."""
        merged_docs, _ = self.rag.retrieve_candidates(query, verbose=False)
        candidates = [self._doc_to_candidate(doc) for doc in merged_docs[:k]]

        # Stage 1: Detection at 3.1 -> 3.2 boundary
        if enable_neardup:
            candidates = self.neardup_detector.filter_candidates(candidates, query=query)
        if enable_outlier:
            candidates = self.outlier_detector.filter_candidates(candidates, query=query)

        # Stage 2: Graph Expansion Gate
        if enable_graph_gate:
            candidates = self.graph_gate.filter_candidates(candidates, query=query)

        # Stage 3: Prompt-Level Sanitization
        if enable_sanitizer:
            candidates = self.sanitizer.filter_candidates(candidates, query=query)

        return [cand for cand in candidates if not cand.quarantined]
