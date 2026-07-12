import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from app import config


@dataclass
class HITLApprovalRequest:
    request_id: str
    action_type: str  # "EXPORT_DOCUMENT", "DELETE_DOCUMENT", "TAG_DOCUMENT"
    target_path: str
    payload_preview: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    status: str = "PENDING_APPROVAL"  # "PENDING_APPROVAL", "APPROVED", "REJECTED", "EXECUTED"


@dataclass
class RiskReport:
    is_safe: bool
    flagged_threats: List[str] = field(default_factory=list)
    audit_stage: str = "INBOUND_PROMPT"


@dataclass
class MissionSnapshot:
    mission_id: str
    goal: str
    synthesized_report: str
    hitl_request: Optional[HITLApprovalRequest] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionResult:
    mission_id: str
    status: str  # "COMPLETED", "WAITING_FOR_HITL", "SECURITY_BLOCKED", "REJECTED", "EXECUTED"
    report: str
    risk_report: Optional[RiskReport] = None
    hitl_request: Optional[HITLApprovalRequest] = None
    execution_path: Optional[str] = None


class SecurityAuditorAgent:
    """Specialized Red-Teaming & Guardrail auditor operating at dual checkpoints."""

    DIRECT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s+override",
        r"delete\s+all\s+notes",
        r"drop\s+database",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
    ]

    INDIRECT_INJECTION_PATTERNS = [
        r"system\s+override:\s+delete",
        r"\[INST\].*delete",
        r"<<SYS>>.*override",
        r"ignore\s+previous\s+instructions\s+and\s+execute",
    ]

    def audit_inbound_prompt(self, prompt: str) -> RiskReport:
        """Checkpoint 1: Audit inbound prompt for direct override/injection attempts."""
        flagged = []
        q_lower = prompt.lower()
        for pat in self.DIRECT_INJECTION_PATTERNS:
            if re.search(pat, q_lower):
                flagged.append(f"Direct Prompt Injection attempt detected matching pattern: '{pat}'")

        return RiskReport(
            is_safe=(len(flagged) == 0),
            flagged_threats=flagged,
            audit_stage="INBOUND_PROMPT",
        )

    def audit_retrieved_chunks(self, chunks: List[Any]) -> RiskReport:
        """Checkpoint 2: Audit retrieved chunks for indirect prompt injection or payload hiding."""
        flagged = []
        for idx, chunk in enumerate(chunks, 1):
            content = getattr(chunk, "content", "") or getattr(chunk, "page_content", "") or str(chunk)
            c_lower = content.lower()
            for pat in self.INDIRECT_INJECTION_PATTERNS:
                if re.search(pat, c_lower):
                    source = getattr(chunk, "source", f"Chunk #{idx}")
                    flagged.append(
                        f"Indirect Prompt Injection payload in '{source}' matching pattern: '{pat}'"
                    )

        return RiskReport(
            is_safe=(len(flagged) == 0),
            flagged_threats=flagged,
            audit_stage="INTERMEDIARY_CHUNKS",
        )


class ResearcherAgent:
    """Specialized agent querying the local document repository."""

    def gather_evidence(self, rag_engine: Any, topic: str) -> List[Any]:
        if not rag_engine or not hasattr(rag_engine, "retrieve_candidates"):
            return []
        candidates, _ = rag_engine.retrieve_candidates(topic)
        return candidates


class SynthesisAgent:
    """Specialized agent compiling grounded findings into a structured report."""

    def synthesize_report(self, goal: str, chunks: List[Any], risk_report: RiskReport) -> str:
        sections = [
            f"# Grounded Mission Report: {goal}\n",
            f"**Security Audit Status**: {'SAFE' if risk_report.is_safe else 'SECURITY ADVISORY'}\n",
        ]
        if chunks:
            sections.append("## Evidence Citations\n")
            for idx, c in enumerate(chunks[:5], 1):
                source = getattr(c, "source", f"Note #{idx}")
                text = getattr(c, "content", "") or getattr(c, "page_content", "") or str(c)
                snippet = text.strip()[:300]
                sections.append(f"### {idx}. `{source}`\n{snippet}...\n")
        else:
            sections.append("No direct local document evidence retrieved.\n")

        sections.append("## Synthesis Summary\n")
        sections.append(
            "Synthesized multi-agent findings grounded in verified local documents."
        )
        return "\n".join(sections)


class LocalActionAgent:
    """Specialized agent executing local file mutations atomically and idempotently."""

    def execute_action(self, hitl_request: HITLApprovalRequest) -> str:
        if hitl_request.status == "EXECUTED":
            return hitl_request.target_path

        if hitl_request.action_type in ("EXPORT_NOTE", "EXPORT_DOCUMENT"):
            target = hitl_request.target_path
            parent_dir = os.path.dirname(target)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            tmp_path = f"{target}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(hitl_request.payload_preview)
            os.replace(tmp_path, target)

            hitl_request.status = "EXECUTED"
            return target

        raise ValueError(f"Unsupported action type: {hitl_request.action_type}")


class MissionOrchestrator:
    """Orchestrates multi-agent specialists with serialized state persistence and HITL gating."""

    def __init__(self, state_dir: Optional[str] = None):
        self.state_dir = state_dir or getattr(config, "MISSION_STATE_DIR", ".mission_state")
        os.makedirs(self.state_dir, exist_ok=True)
        self.security_auditor = SecurityAuditorAgent()
        self.researcher = ResearcherAgent()
        self.synthesizer = SynthesisAgent()
        self.action_agent = LocalActionAgent()

    def _state_file(self, request_id: str) -> str:
        return os.path.join(self.state_dir, f"{request_id}.json")

    def _save_snapshot(self, snapshot: MissionSnapshot):
        data = asdict(snapshot)
        with open(self._state_file(snapshot.mission_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_snapshot(self, request_id: str) -> Optional[MissionSnapshot]:
        fpath = self._state_file(request_id)
        if not os.path.exists(fpath):
            return None
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        hitl_raw = raw.get("hitl_request")
        hitl_req = HITLApprovalRequest(**hitl_raw) if hitl_raw else None
        return MissionSnapshot(
            mission_id=raw["mission_id"],
            goal=raw["goal"],
            synthesized_report=raw["synthesized_report"],
            hitl_request=hitl_req,
            metadata=raw.get("metadata", {}),
        )

    def run_mission(
        self,
        mission_id: str,
        goal: str,
        rag_engine: Any = None,
        requested_action: Optional[Dict[str, str]] = None,
    ) -> MissionResult:
        """Execute a multi-agent mission with dual-checkpoint guardrails and HITL gating."""
        # Checkpoint 1: Inbound prompt audit
        inbound_audit = self.security_auditor.audit_inbound_prompt(goal)
        if not inbound_audit.is_safe:
            return MissionResult(
                mission_id=mission_id,
                status="SECURITY_BLOCKED",
                report="\n".join(inbound_audit.flagged_threats),
                risk_report=inbound_audit,
            )

        # Research phase
        chunks = self.researcher.gather_evidence(rag_engine, goal)

        # Checkpoint 2: Intermediary chunk audit
        chunk_audit = self.security_auditor.audit_retrieved_chunks(chunks)
        if not chunk_audit.is_safe:
            return MissionResult(
                mission_id=mission_id,
                status="SECURITY_BLOCKED",
                report="\n".join(chunk_audit.flagged_threats),
                risk_report=chunk_audit,
            )

        # Synthesis phase
        report = self.synthesizer.synthesize_report(goal, chunks, chunk_audit)

        # Check if local action mutation requested
        if requested_action:
            hitl_req = HITLApprovalRequest(
                request_id=mission_id,
                action_type=requested_action.get("action_type", "EXPORT_NOTE"),
                target_path=requested_action.get("target_path", "export.md"),
                payload_preview=report,
                risk_level=requested_action.get("risk_level", "MEDIUM"),
                status="PENDING_APPROVAL",
            )
            snapshot = MissionSnapshot(
                mission_id=mission_id,
                goal=goal,
                synthesized_report=report,
                hitl_request=hitl_req,
                metadata={"status": "PENDING_APPROVAL"},
            )
            self._save_snapshot(snapshot)
            return MissionResult(
                mission_id=mission_id,
                status="WAITING_FOR_HITL",
                report=report,
                risk_report=chunk_audit,
                hitl_request=hitl_req,
            )

        return MissionResult(
            mission_id=mission_id,
            status="COMPLETED",
            report=report,
            risk_report=chunk_audit,
        )

    def approve_action(self, request_id: str) -> MissionResult:
        """Approve and execute a suspended HITL mission idempotently."""
        snapshot = self._load_snapshot(request_id)
        if not snapshot or not snapshot.hitl_request:
            raise ValueError(f"No pending mission found with ID: {request_id}")

        req = snapshot.hitl_request
        if req.status == "EXECUTED":
            return MissionResult(
                mission_id=snapshot.mission_id,
                status="EXECUTED",
                report=snapshot.synthesized_report,
                hitl_request=req,
                execution_path=req.target_path,
            )

        exec_path = self.action_agent.execute_action(req)
        snapshot.hitl_request.status = "EXECUTED"
        self._save_snapshot(snapshot)

        return MissionResult(
            mission_id=snapshot.mission_id,
            status="EXECUTED",
            report=snapshot.synthesized_report,
            hitl_request=req,
            execution_path=exec_path,
        )

    def reject_action(self, request_id: str) -> MissionResult:
        """Reject a pending HITL action."""
        snapshot = self._load_snapshot(request_id)
        if not snapshot or not snapshot.hitl_request:
            raise ValueError(f"No pending mission found with ID: {request_id}")

        snapshot.hitl_request.status = "REJECTED"
        self._save_snapshot(snapshot)

        return MissionResult(
            mission_id=snapshot.mission_id,
            status="REJECTED",
            report="Human operator rejected the requested action.",
            hitl_request=snapshot.hitl_request,
        )
