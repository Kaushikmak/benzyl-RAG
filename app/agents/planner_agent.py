"""PlannerAgent responsible for analyzing query to generate a structured retrieval plan."""

from typing import Any, Dict


class PlannerAgent:
    """Analyzes incoming request to determine intent, retrieval strategy, filters, and top-k."""

    def create_retrieval_plan(self, query: str) -> Dict[str, Any]:
        """Generate execution retrieval plan for a user query."""
        q_lower = query.lower()

        intent = "GENERAL_QUERY"
        if any(w in q_lower for w in ["compare", "difference", "vs"]):
            intent = "COMPARISON"
        elif any(w in q_lower for w in ["summarize", "summary", "overview"]):
            intent = "SUMMARIZATION"
        elif any(w in q_lower for w in ["how to", "steps", "guide", "tutorial"]):
            intent = "PROCEDURAL"

        search_mode = "HYBRID"
        if "exact" in q_lower or "quote" in q_lower:
            search_mode = "DENSE_EXACT"

        top_k = 10
        if intent == "SUMMARIZATION":
            top_k = 15

        return {
            "intent": intent,
            "retrieval_strategy": "MULTI_STAGE_FUNNEL",
            "metadata_filters": {},
            "top_k": top_k,
            "search_mode": search_mode,
            "expected_answer_type": "MARKDOWN_REPORT",
        }
