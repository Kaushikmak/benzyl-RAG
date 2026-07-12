"""Semantic Query Router delegating to modular agents in `app.agents`."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.agents.math_agent import eval_math_ast, SafeMathEvaluator
from app.agents.orchestrator import AgentOrchestrator, AgentDecision


@dataclass
class RouteDecision:
    action: str
    confidence: float
    reason: str
    direct_answer: Optional[str] = None
    file_action: Optional[str] = None
    target_file: Optional[str] = None
    core_query: Optional[str] = None


class QueryRouter:
    """High-speed semantic query router delegating to specialized domain agents."""

    def __init__(self, workspace_root: str = "."):
        self.orchestrator = AgentOrchestrator(workspace_root=workspace_root)

    def triage(
        self, query: str, system_context: Optional[Dict[str, Any]] = None
    ) -> RouteDecision:
        dec: AgentDecision = self.orchestrator.triage(query, system_context)
        return RouteDecision(
            action=dec.action,
            confidence=dec.confidence,
            reason=dec.reason,
            direct_answer=dec.direct_answer,
            file_action=dec.file_action,
            target_file=dec.target_file,
            core_query=dec.core_query,
        )


__all__ = [
    "RouteDecision",
    "QueryRouter",
    "eval_math_ast",
    "SafeMathEvaluator",
]
