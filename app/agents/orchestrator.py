"""AgentOrchestrator central workflow engine for Layer 5 Enterprise Multi-Agent Architecture (`benzene-rag`)."""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app import config
from app.agents.cache_agent import CacheAgent
from app.agents.citation_agent import CitationAgent
from app.agents.compression_agent import CompressionAgent
from app.agents.exceptions import (
    MissionAlreadyExecuted,
    MissionNotFound,
    MissionRejected,
    SecurityException,
)
from app.agents.file_agent import FileAgent
from app.agents.formatter_agent import FormatterAgent
from app.agents.math_agent import MathAgent
from app.agents.models import (
    Citation,
    HITLApprovalRequest,
    HITLStatus,
    MissionMetrics,
    MissionSnapshot,
    VerificationReport,
)
from app.agents.observability_agent import ObservabilityAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.reflection_agent import ReflectionAgent
from app.agents.reranker_agent import RerankerAgent
from app.agents.researcher_agent import ResearcherAgent
from app.agents.rewrite_agent import RewriteAgent
from app.agents.security_agent import SecurityAgent
from app.agents.status_agent import StatusAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.agents.verification_agent import VerificationAgent
from app.agents.utils import load_snapshot, save_snapshot, update_snapshot

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    action: str  # e.g. "DIRECT_MATH", "DIRECT_STATUS", "DIRECT_CONVERSATION", "FILE_ACTION", "DOCUMENT_RAG"
    confidence: float
    reason: str
    direct_answer: Optional[str] = None
    file_action: Optional[str] = None
    target_file: Optional[str] = None
    core_query: Optional[str] = None


class AgentOrchestrator:
    """Enterprise multi-agent orchestrator owning all agent lifecycles, execution workflow, security, and HITL approvals."""

    def __init__(
        self,
        workspace_root: str = ".",
        retriever_callback: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
        reranker_callback: Optional[Callable[[str, List[Dict[str, Any]], int], List[Dict[str, Any]]]] = None,
        llm_callback: Optional[Callable[[str, List[Dict[str, Any]], List[Citation]], str]] = None,
        base_state_dir: str = ".mission_state",
    ):
        self.workspace_root = workspace_root
        self.base_state_dir = base_state_dir

        # Instantiate all 16 single-responsibility agents internally
        self.security_agent = SecurityAgent()
        self.planner_agent = PlannerAgent()
        self.rewrite_agent = RewriteAgent()
        self.cache_agent = CacheAgent()
        self.researcher_agent = ResearcherAgent(retriever_callback=retriever_callback)
        self.compression_agent = CompressionAgent()
        self.reranker_agent = RerankerAgent(reranker_callback=reranker_callback)
        self.citation_agent = CitationAgent()
        self.synthesis_agent = SynthesisAgent(llm_callback=llm_callback)
        self.reflection_agent = ReflectionAgent()
        self.verification_agent = VerificationAgent()
        self.formatter_agent = FormatterAgent()
        self.observability_agent = ObservabilityAgent()
        self.file_agent = FileAgent(workspace_root=workspace_root)
        self.math_agent = MathAgent()
        self.status_agent = StatusAgent()

        self._conversational_greetings = {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "thank you",
            "thanks",
            "who are you",
            "what are you",
        }

    def _try_conversational_route(self, query: str) -> Optional[AgentDecision]:
        import re

        q_norm = re.sub(r"[^\w\s]", "", query.strip().lower())
        if q_norm in self._conversational_greetings:
            if "who" in q_norm or "what" in q_norm:
                ans = (
                    "I am **benzene-rag**, an enterprise local-first RAG assistant "
                    "equipped with a 16-agent Layer 5 Orchestration Engine."
                )
            elif "thank" in q_norm:
                ans = "You're welcome! Let me know if you need anything else."
            else:
                ans = "Hello! I am ready to assist with enterprise RAG retrieval and analysis."
            return AgentDecision(
                action="DIRECT_CONVERSATION",
                confidence=0.98,
                reason="Conversational greeting routed directly.",
                direct_answer=ans,
            )
        return None

    def triage(
        self, query: str, system_context: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """Backward-compatible query triage across agents."""
        if not getattr(config, "ENABLE_QUERY_ROUTING", True):
            return AgentDecision(
                action="DOCUMENT_RAG",
                confidence=1.0,
                reason="Routing disabled via config.",
            )

        file_intent = self.file_agent.parse_intent(query)
        if file_intent:
            action_type = file_intent["action"]
            target_file = file_intent["target_file"]
            core_q = file_intent["core_query"]
            if action_type == "DELETE":
                report = self.file_agent.execute("DELETE", target_file)
                return AgentDecision(
                    action="FILE_ACTION",
                    confidence=1.0,
                    reason=f"File deletion handled by FileAgent for {target_file}.",
                    direct_answer=report,
                    file_action="DELETE",
                    target_file=target_file,
                )
            else:
                return AgentDecision(
                    action="FILE_ACTION",
                    confidence=1.0,
                    reason=f"File save action handled by FileAgent for {target_file}.",
                    file_action=action_type,
                    target_file=target_file,
                    core_query=core_q,
                )

        math_answer = self.math_agent.evaluate(query)
        if math_answer:
            return AgentDecision(
                action="DIRECT_MATH",
                confidence=1.0,
                reason="Pure arithmetic expression evaluated by MathAgent.",
                direct_answer=math_answer,
            )

        status_answer = self.status_agent.try_status_route(query, system_context)
        if status_answer:
            return AgentDecision(
                action="DIRECT_STATUS",
                confidence=0.99,
                reason="System telemetry query handled by StatusAgent.",
                direct_answer=status_answer,
            )

        conv_decision = self._try_conversational_route(query)
        if conv_decision:
            return conv_decision

        return AgentDecision(
            action="DOCUMENT_RAG",
            confidence=0.95,
            reason="Substantive query routed to full RAG retrieval funnel.",
        )

    def run_mission(
        self,
        query: str,
        request_id: Optional[str] = None,
        target_format: str = "Markdown",
        require_hitl_for_files: bool = True,
    ) -> MissionSnapshot:
        """Execute the 14-stage multi-agent orchestration pipeline."""
        req_id = request_id or uuid.uuid4().hex
        timestamps: Dict[str, float] = {"start": time.perf_counter()}

        logger.info("Mission %s started for query: '%s'", req_id, query)

        # Stage 1: SecurityAgent inbound inspection
        risk_report = self.security_agent.audit_inbound_prompt(query)
        if not risk_report.is_safe:
            logger.error("Mission %s aborted by SecurityAgent: %s", req_id, risk_report.reason)
            raise SecurityException(f"Inbound prompt blocked: {risk_report.reason}")

        # Stage 2: PlannerAgent
        file_intent = self.file_agent.parse_intent(query)
        effective_query = file_intent["core_query"] if file_intent and file_intent.get("core_query") else query
        retrieval_plan = self.planner_agent.create_retrieval_plan(effective_query)

        # Stage 3: RewriteAgent
        rewritten_query = self.rewrite_agent.rewrite(effective_query, retrieval_plan)

        # Stage 4: CacheAgent check
        cache_key = f"mission:{rewritten_query}|{retrieval_plan.get('top_k')}"
        cached_report = self.cache_agent.get(cache_key)
        if cached_report:
            timestamps["end"] = time.perf_counter()
            metrics = self.observability_agent.record_metrics(
                timestamps, chunk_count=0, cache_hit=True
            )
            logger.info("Mission %s hit CacheAgent", req_id)
            return MissionSnapshot(
                request_id=req_id,
                original_query=query,
                rewritten_query=rewritten_query,
                retrieval_plan=retrieval_plan,
                synthesized_report=str(cached_report),
                metrics=metrics,
                timestamps=timestamps,
            )

        # Stage 5: ResearcherAgent retrieval
        raw_chunks = self.researcher_agent.retrieve(rewritten_query, retrieval_plan)
        timestamps["retrieved"] = time.perf_counter()

        # Stage 6: SecurityAgent chunk audit
        safe_chunks, quarantined_chunks = self.security_agent.audit_retrieved_chunks(raw_chunks)

        # Stage 7: CompressionAgent deduplication
        compressed_chunks = self.compression_agent.compress(safe_chunks)

        # Stage 8: RerankerAgent
        top_k = retrieval_plan.get("top_k", 5)
        reranked_chunks = self.reranker_agent.rerank(rewritten_query, compressed_chunks, top_k=top_k)
        timestamps["reranked"] = time.perf_counter()

        # Stage 9: CitationAgent extraction
        citations = self.citation_agent.extract_citations(reranked_chunks)

        # Stage 10: SynthesisAgent
        synthesized_report = self.synthesis_agent.synthesize(rewritten_query, reranked_chunks, citations)
        timestamps["synthesized"] = time.perf_counter()

        # Stage 11: ReflectionAgent audit
        reflection_data = self.reflection_agent.reflect(rewritten_query, synthesized_report, reranked_chunks)
        if reflection_data.get("needs_retry"):
            logger.warning("Mission %s ReflectionAgent requested retry logic", req_id)

        # Stage 12: VerificationAgent final gate
        verification_report = self.verification_agent.verify(rewritten_query, synthesized_report, citations)

        # Stage 13: FormatterAgent
        formatted_report = self.formatter_agent.format(synthesized_report, target_format=target_format)
        self.cache_agent.put(cache_key, formatted_report)

        timestamps["end"] = time.perf_counter()

        # Stage 14: ObservabilityAgent metrics
        approval_req_cnt = 1 if (file_intent and require_hitl_for_files) else 0
        metrics = self.observability_agent.record_metrics(
            timestamps,
            chunk_count=len(reranked_chunks),
            cache_hit=False,
            security_violations=len(quarantined_chunks),
            approval_requests=approval_req_cnt,
            failures=0,
        )

        hitl_request: Optional[HITLApprovalRequest] = None
        if file_intent and require_hitl_for_files:
            action_type = file_intent["action"]
            target_file = file_intent["target_file"]
            self.file_agent.validate_target_path(target_file)
            hitl_request = HITLApprovalRequest(
                request_id=req_id,
                action_type=action_type,
                payload={"target_file": target_file, "content": formatted_report},
                status=HITLStatus.PENDING,
            )

        snapshot = MissionSnapshot(
            request_id=req_id,
            original_query=query,
            rewritten_query=rewritten_query,
            retrieval_plan=retrieval_plan,
            retrieved_chunks=raw_chunks,
            compressed_chunks=compressed_chunks,
            reranked_chunks=reranked_chunks,
            synthesized_report=formatted_report,
            citations=citations,
            verification_report=verification_report,
            metrics=metrics,
            hitl_request=hitl_request,
            timestamps=timestamps,
        )

        if hitl_request:
            save_snapshot(snapshot, base_dir=self.base_state_dir)
            logger.info("Mission %s requires HITL approval. Saved snapshot to %s", req_id, self.base_state_dir)

        return snapshot

    def approve_action(self, request_id: str) -> MissionSnapshot:
        """Approve Human-in-the-Loop request safely and idempotently executing the pending local action."""
        snapshot = load_snapshot(request_id, base_dir=self.base_state_dir)
        if not snapshot.hitl_request:
            raise MissionNotFound(f"No HITL request attached to mission {request_id}")

        if snapshot.hitl_request.status == HITLStatus.EXECUTED:
            raise MissionAlreadyExecuted(f"Mission {request_id} action has already been EXECUTED.")
        if snapshot.hitl_request.status == HITLStatus.REJECTED:
            raise MissionRejected(f"Mission {request_id} was previously REJECTED.")

        payload = snapshot.hitl_request.payload
        action_type = snapshot.hitl_request.action_type
        target_file = payload.get("target_file")
        content = payload.get("content", snapshot.synthesized_report)

        logger.info("Approving HITL action %s for mission %s on target %s", action_type, request_id, target_file)
        execution_report = self.file_agent.execute(action_type, target_file, content=content)

        snapshot.hitl_request.status = HITLStatus.EXECUTED
        snapshot.metadata["execution_report"] = execution_report
        update_snapshot(snapshot, base_dir=self.base_state_dir)

        return snapshot

    def reject_action(self, request_id: str, reason: str = "Rejected by user") -> MissionSnapshot:
        """Reject pending Human-in-the-Loop request."""
        snapshot = load_snapshot(request_id, base_dir=self.base_state_dir)
        if not snapshot.hitl_request:
            raise MissionNotFound(f"No HITL request attached to mission {request_id}")

        if snapshot.hitl_request.status == HITLStatus.EXECUTED:
            raise MissionAlreadyExecuted(f"Cannot reject mission {request_id} as it has already EXECUTED.")

        snapshot.hitl_request.status = HITLStatus.REJECTED
        snapshot.metadata["rejection_reason"] = reason
        update_snapshot(snapshot, base_dir=self.base_state_dir)
        logger.info("Rejected HITL action for mission %s (%s)", request_id, reason)
        return snapshot
