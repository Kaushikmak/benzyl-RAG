"""Shared graph state definitions for LangGraph multi-agent orchestration (`app.agents.state`)."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from app.agents.models import Citation, HITLApprovalRequest, MissionMetrics, VerificationReport


class MissionState(TypedDict, total=False):
    """Shared state dictionary passed across LangGraph nodes in the AgentOrchestrator graph."""
    query: str
    request_id: str
    target_format: str
    require_hitl_for_files: bool
    retrieval_plan: Dict[str, Any]
    effective_query: str
    rewritten_query: str
    
    # Parallel retrieval branches and joined raw chunks
    vector_chunks: List[Any]
    bm25_chunks: List[Any]
    graph_chunks: List[Any]
    raw_chunks: List[Any]
    quarantined_chunks: List[Any]
    
    compressed_chunks: List[Any]
    reranked_chunks: List[Any]
    citations: List[Citation]
    synthesized_report: str
    verification_report: Optional[VerificationReport]
    metrics: MissionMetrics
    hitl_request: Optional[HITLApprovalRequest]
    timestamps: Dict[str, float]
    metadata: Dict[str, Any]
    
    # DeepAgent cyclic self-correction & routing state
    retry_count: int
    needs_retry: bool
    is_safe: bool
    security_reason: Optional[str]
    file_intent: Optional[Dict[str, Any]]
    cached_report: Optional[str]
