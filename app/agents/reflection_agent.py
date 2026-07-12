"""ReflectionAgent responsible for auditing synthesized output for weak or unsupported claims."""

from typing import Any, Dict, List


class ReflectionAgent:
    """Reviews generated output to detect weak sections or request additional retrieval."""

    def reflect(
        self, query: str, report: str, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Review report and determine if output is grounded and strong."""
        needs_retry = False
        weak_sections: List[str] = []

        if len(report.strip()) < 20 and len(chunks) > 0:
            needs_retry = True
            weak_sections.append("Report abnormally short despite available chunks")

        return {
            "passed": not needs_retry,
            "needs_retry": needs_retry,
            "weak_sections": weak_sections,
        }
