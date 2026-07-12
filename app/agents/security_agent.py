"""SecurityAgent responsible for inbound prompt inspection and retrieved chunk auditing."""

import logging
import re
from typing import Any, Dict, List, Tuple
from app.agents.models import RiskLevel, RiskReport

logger = logging.getLogger(__name__)


class SecurityAgent:
    """Security guardrail auditing inbound prompts and quarantining dangerous chunks."""

    def __init__(self):
        self._prompt_injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s+override",
            r"you\s+are\s+now\s+in\s+developer\s+mode",
            r"do\s+anything\s+now",
            r"reveal\s+system\s+prompt",
            r"bypass\s+guardrails",
        ]

    def audit_inbound_prompt(self, query: str) -> RiskReport:
        """Inspect inbound prompt for injection, jailbreak, or malicious tool requests."""
        q_lower = query.lower()
        for pattern in self._prompt_injection_patterns:
            if re.search(pattern, q_lower):
                logger.warning("SecurityAgent detected inbound prompt injection pattern: %s", pattern)
                return RiskReport(
                    is_safe=False,
                    reason=f"Prompt injection or jailbreak attempt detected ({pattern})",
                    risk_level=RiskLevel.CRITICAL,
                )
        return RiskReport(is_safe=True, reason="Inbound prompt safe", risk_level=RiskLevel.LOW)

    def audit_retrieved_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Audit retrieved chunks and quarantine any containing adversarial prompt injections."""
        safe_chunks: List[Dict[str, Any]] = []
        quarantined_chunks: List[Dict[str, Any]] = []

        for chunk in chunks:
            content = str(chunk.get("content", "")).lower()
            is_compromised = False
            for pattern in self._prompt_injection_patterns:
                if re.search(pattern, content):
                    is_compromised = True
                    logger.warning(
                        "SecurityAgent quarantined chunk '%s' due to pattern: %s",
                        chunk.get("id", "unknown"),
                        pattern,
                    )
                    break

            if is_compromised:
                quarantined_chunks.append(chunk)
            else:
                safe_chunks.append(chunk)

        return safe_chunks, quarantined_chunks
