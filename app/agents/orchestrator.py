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
from app.agents.state import MissionState
from app.agents.graph import build_agent_graph

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

        agents_dict = {
            "security": self.security_agent,
            "planner": self.planner_agent,
            "rewrite": self.rewrite_agent,
            "cache": self.cache_agent,
            "researcher": self.researcher_agent,
            "compression": self.compression_agent,
            "reranker": self.reranker_agent,
            "citation": self.citation_agent,
            "synthesis": self.synthesis_agent,
            "reflection": self.reflection_agent,
            "verification": self.verification_agent,
            "formatter": self.formatter_agent,
            "observability": self.observability_agent,
            "file": self.file_agent,
            "math": self.math_agent,
            "status": self.status_agent,
        }
        self.graph = build_agent_graph(agents_dict)

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
        """Execute the multi-agent orchestration pipeline using LangGraph StateGraph."""
        req_id = request_id or uuid.uuid4().hex
        timestamps: Dict[str, float] = {"start": time.perf_counter()}
        logger.info("Mission %s started for query: '%s'", req_id, query)

        initial_state: MissionState = {
            "query": query,
            "request_id": req_id,
            "target_format": target_format,
            "require_hitl_for_files": require_hitl_for_files,
            "timestamps": timestamps,
            "retry_count": 0,
        }

        config = {"configurable": {"thread_id": req_id}}
        final_state = self.graph.invoke(initial_state, config=config)

        if not final_state.get("is_safe", True):
            reason = final_state.get("security_reason", "Inbound prompt blocked")
            logger.error("Mission %s aborted by SecurityAgent: %s", req_id, reason)
            raise SecurityException(f"Inbound prompt blocked: {reason}")

        hitl_request = final_state.get("hitl_request")
        metrics = final_state.get("metrics") or MissionMetrics()
        verification_report = final_state.get("verification_report")

        snapshot = MissionSnapshot(
            request_id=req_id,
            original_query=query,
            rewritten_query=final_state.get("rewritten_query", ""),
            retrieval_plan=final_state.get("retrieval_plan", {}),
            retrieved_chunks=final_state.get("raw_chunks", []),
            compressed_chunks=final_state.get("compressed_chunks", []),
            reranked_chunks=final_state.get("reranked_chunks", []),
            synthesized_report=final_state.get("synthesized_report", ""),
            citations=final_state.get("citations", []),
            verification_report=verification_report,
            metrics=metrics,
            hitl_request=hitl_request,
            timestamps=final_state.get("timestamps", timestamps),
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

        try:
            from langgraph.types import Command
            self.graph.invoke(Command(resume=True), config={"configurable": {"thread_id": request_id}})
        except Exception as exc:
            logger.debug("Checkpointer thread %s resume skipped or already completed: %s", request_id, exc)

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
