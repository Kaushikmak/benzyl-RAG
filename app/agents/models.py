"""Shared Pydantic v2 data models and Enums for the Enterprise Multi-Agent Orchestration Engine."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HITLStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"


class RiskReport(BaseModel):
    is_safe: bool = Field(default=True, description="Whether the prompt/payload is safe to process")
    reason: str = Field(default="No security violations detected", description="Reason for risk level")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Assessed risk level")


class HITLApprovalRequest(BaseModel):
    request_id: str = Field(..., description="Unique UUID for the HITL request")
    action_type: str = Field(..., description="Action category requiring approval (e.g., FILE_SAVE, EXPORT)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Parameters and payload for the action")
    status: HITLStatus = Field(default=HITLStatus.PENDING, description="Current approval status")


class Citation(BaseModel):
    source: str = Field(..., description="Document source filename or path")
    snippet: str = Field(..., description="Grounded excerpt supporting the statement")
    relevance_score: float = Field(default=1.0, description="Confidence/relevance score of the citation")


class VerificationReport(BaseModel):
    passed: bool = Field(default=True, description="Whether the output passed verification")
    citations_valid: bool = Field(default=True, description="Whether all citations are grounded")
    completeness_score: float = Field(default=1.0, description="Completeness evaluation score (0.0 - 1.0)")
    confidence_score: float = Field(default=1.0, description="Overall verification confidence score (0.0 - 1.0)")
    unsupported_claims: List[str] = Field(default_factory=list, description="Any ungrounded claims detected")


class MissionMetrics(BaseModel):
    retrieval_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")
    reranking_ms: float = Field(default=0.0, description="Reranking latency in milliseconds")
    synthesis_ms: float = Field(default=0.0, description="Synthesis latency in milliseconds")
    total_ms: float = Field(default=0.0, description="Total pipeline latency in milliseconds")
    chunk_count: int = Field(default=0, description="Number of chunks retrieved")
    cache_hit: bool = Field(default=False, description="Whether the query hit the cache")
    security_violations: int = Field(default=0, description="Number of security violations/quarantined chunks")
    approval_requests: int = Field(default=0, description="Number of HITL approval requests generated")
    failures: int = Field(default=0, description="Number of pipeline step failures")


class MissionSnapshot(BaseModel):
    request_id: str = Field(..., description="Unique mission execution identifier")
    original_query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(default="", description="Optimized query produced by RewriteAgent")
    retrieval_plan: Dict[str, Any] = Field(default_factory=dict, description="Execution plan from PlannerAgent")
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Raw retrieved chunks")
    compressed_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Compressed/deduplicated chunks")
    reranked_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Reranked chunks")
    synthesized_report: str = Field(default="", description="Generated Markdown report")
    citations: List[Citation] = Field(default_factory=list, description="Extracted citations")
    verification_report: Optional[VerificationReport] = Field(default=None, description="Report from VerificationAgent")
    metrics: MissionMetrics = Field(default_factory=MissionMetrics, description="Observability metrics")
    hitl_request: Optional[HITLApprovalRequest] = Field(default=None, description="HITL approval state if required")
    timestamps: Dict[str, float] = Field(default_factory=dict, description="Stage timestamps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional mission metadata")
