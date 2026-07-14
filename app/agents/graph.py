"""LangGraph StateGraph builder for Enterprise Multi-Agent Architecture (`app.agents.graph`)."""

import logging
import time
from typing import Any, Dict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import MissionState
from app.agents.models import HITLApprovalRequest, HITLStatus
from app.agents.exceptions import SecurityException

logger = logging.getLogger(__name__)


def build_agent_graph(agents: Dict[str, Any], checkpointer: Any = None) -> Any:
    """Build and compile the multi-agent StateGraph with parallel retrieval and DeepAgent self-correction."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(MissionState)

    # ── Node Definitions ────────────────────────────────────────────────────────
    def security_inbound_node(state: MissionState) -> Dict[str, Any]:
        query = state.get("query", "")
        risk_report = agents["security"].audit_inbound_prompt(query)
        file_intent = agents["file"].parse_intent(query)
        effective_query = file_intent["core_query"] if file_intent and file_intent.get("core_query") else query
        return {
            "is_safe": risk_report.is_safe,
            "security_reason": risk_report.reason,
            "file_intent": file_intent,
            "effective_query": effective_query,
            "retry_count": state.get("retry_count", 0),
            "timestamps": {**state.get("timestamps", {}), "inbound_checked": time.perf_counter()},
        }

    def security_block_node(state: MissionState) -> Dict[str, Any]:
        reason = state.get("security_reason", "Security violation detected")
        logger.error("Mission aborted by SecurityAgent: %s", reason)
        timestamps = {**state.get("timestamps", {}), "end": time.perf_counter()}
        return {"timestamps": timestamps}

    def planner_node(state: MissionState) -> Dict[str, Any]:
        effective_query = state.get("effective_query", state.get("query", ""))
        retrieval_plan = agents["planner"].create_retrieval_plan(effective_query)
        return {"retrieval_plan": retrieval_plan}

    def rewrite_node(state: MissionState) -> Dict[str, Any]:
        effective_query = state.get("effective_query", state.get("query", ""))
        retrieval_plan = state.get("retrieval_plan", {})
        rewritten_query = agents["rewrite"].rewrite(effective_query, retrieval_plan)
        return {"rewritten_query": rewritten_query}

    def cache_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        retrieval_plan = state.get("retrieval_plan", {})
        cache_key = f"mission:{rewritten_query}|{retrieval_plan.get('top_k')}"
        cached_report = agents["cache"].get(cache_key)
        return {"cached_report": cached_report}

    # Parallel retrieval branches
    def vector_retrieval_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        retrieval_plan = state.get("retrieval_plan", {})
        vector_plan = {**retrieval_plan, "branch": "vector"}
        chunks = agents["researcher"].retrieve(rewritten_query, vector_plan)
        return {"vector_chunks": chunks}

    def bm25_retrieval_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        retrieval_plan = state.get("retrieval_plan", {})
        bm25_plan = {**retrieval_plan, "branch": "bm25"}
        chunks = agents["researcher"].retrieve(rewritten_query, bm25_plan)
        return {"bm25_chunks": chunks}

    def graph_retrieval_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        retrieval_plan = state.get("retrieval_plan", {})
        graph_plan = {**retrieval_plan, "branch": "graph"}
        chunks = agents["researcher"].retrieve(rewritten_query, graph_plan)
        return {"graph_chunks": chunks}

    def research_join_node(state: MissionState) -> Dict[str, Any]:
        vc = state.get("vector_chunks", []) or []
        bc = state.get("bm25_chunks", []) or []
        gc = state.get("graph_chunks", []) or []
        combined = vc + bc + gc
        timestamps = {**state.get("timestamps", {}), "retrieved": time.perf_counter()}
        return {"raw_chunks": combined, "timestamps": timestamps}

    def security_chunk_node(state: MissionState) -> Dict[str, Any]:
        raw_chunks = state.get("raw_chunks", [])
        safe_chunks, quarantined_chunks = agents["security"].audit_retrieved_chunks(raw_chunks)
        return {"raw_chunks": safe_chunks, "quarantined_chunks": quarantined_chunks}

    def compression_node(state: MissionState) -> Dict[str, Any]:
        safe_chunks = state.get("raw_chunks", [])
        compressed_chunks = agents["compression"].compress(safe_chunks)
        return {"compressed_chunks": compressed_chunks}

    def reranker_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        compressed_chunks = state.get("compressed_chunks", [])
        retrieval_plan = state.get("retrieval_plan", {})
        top_k = retrieval_plan.get("top_k", 5)
        reranked_chunks = agents["reranker"].rerank(rewritten_query, compressed_chunks, top_k=top_k)
        timestamps = {**state.get("timestamps", {}), "reranked": time.perf_counter()}
        return {"reranked_chunks": reranked_chunks, "timestamps": timestamps}

    def citation_node(state: MissionState) -> Dict[str, Any]:
        reranked_chunks = state.get("reranked_chunks", [])
        citations = agents["citation"].extract_citations(reranked_chunks)
        return {"citations": citations}

    def synthesis_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        reranked_chunks = state.get("reranked_chunks", [])
        citations = state.get("citations", [])
        synthesized_report = agents["synthesis"].synthesize(rewritten_query, reranked_chunks, citations)
        timestamps = {**state.get("timestamps", {}), "synthesized": time.perf_counter()}
        return {"synthesized_report": synthesized_report, "timestamps": timestamps}

    def reflection_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        synthesized_report = state.get("synthesized_report", "")
        reranked_chunks = state.get("reranked_chunks", [])
        reflection_data = agents["reflection"].reflect(rewritten_query, synthesized_report, reranked_chunks)
        needs_retry = reflection_data.get("needs_retry", False)
        return {"needs_retry": needs_retry}

    def verification_node(state: MissionState) -> Dict[str, Any]:
        rewritten_query = state.get("rewritten_query", "")
        synthesized_report = state.get("synthesized_report", "")
        citations = state.get("citations", [])
        verification_report = agents["verification"].verify(rewritten_query, synthesized_report, citations)
        
        needs_retry = state.get("needs_retry", False)
        if not verification_report.passed or not verification_report.citations_valid:
            needs_retry = True

        retry_count = state.get("retry_count", 0)
        if needs_retry and retry_count < 2:
            logger.info("Verification/Reflection self-correction triggered (retry %d of 2)", retry_count + 1)
            return {
                "verification_report": verification_report,
                "needs_retry": True,
                "retry_count": retry_count + 1,
            }
        else:
            return {
                "verification_report": verification_report,
                "needs_retry": False,
            }

    def formatter_node(state: MissionState) -> Dict[str, Any]:
        synthesized_report = state.get("synthesized_report", "")
        target_format = state.get("target_format", "Markdown")
        formatted_report = agents["formatter"].format(synthesized_report, target_format=target_format)
        
        rewritten_query = state.get("rewritten_query", "")
        retrieval_plan = state.get("retrieval_plan", {})
        cache_key = f"mission:{rewritten_query}|{retrieval_plan.get('top_k')}"
        agents["cache"].put(cache_key, formatted_report)
        return {"synthesized_report": formatted_report}

    def hitl_check_node(state: MissionState) -> Dict[str, Any]:
        file_intent = state.get("file_intent")
        require_hitl = state.get("require_hitl_for_files", True)
        if file_intent and require_hitl:
            action_type = file_intent["action"]
            target_file = file_intent["target_file"]
            agents["file"].validate_target_path(target_file)
            hitl_request = HITLApprovalRequest(
                request_id=state.get("request_id", ""),
                action_type=action_type,
                payload={"target_file": target_file, "content": state.get("synthesized_report", "")},
                status=HITLStatus.PENDING,
            )
            return {"hitl_request": hitl_request}
        return {"hitl_request": None}

    def hitl_pause_node(state: MissionState) -> Dict[str, Any]:
        # This node is the interrupt boundary for pending Human-in-the-Loop approvals
        return {}

    def observability_node(state: MissionState) -> Dict[str, Any]:
        timestamps = {**state.get("timestamps", {}), "end": time.perf_counter()}
        reranked_chunks = state.get("reranked_chunks", [])
        quarantined_chunks = state.get("quarantined_chunks", [])
        hitl_request = state.get("hitl_request")
        cached_report = state.get("cached_report")
        is_cache_hit = cached_report is not None

        approval_req_cnt = 1 if hitl_request else 0
        metrics = agents["observability"].record_metrics(
            timestamps,
            chunk_count=len(reranked_chunks),
            cache_hit=is_cache_hit,
            security_violations=len(quarantined_chunks),
            approval_requests=approval_req_cnt,
            failures=0 if state.get("is_safe", True) else 1,
        )
        if is_cache_hit and not state.get("synthesized_report"):
            return {"synthesized_report": str(cached_report), "metrics": metrics, "timestamps": timestamps}
        return {"metrics": metrics, "timestamps": timestamps}

    # ── Add Nodes to Graph ──────────────────────────────────────────────────────
    graph.add_node("security_inbound_node", security_inbound_node)
    graph.add_node("security_block_node", security_block_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("cache_node", cache_node)
    graph.add_node("vector_retrieval_node", vector_retrieval_node)
    graph.add_node("bm25_retrieval_node", bm25_retrieval_node)
    graph.add_node("graph_retrieval_node", graph_retrieval_node)
    graph.add_node("research_join_node", research_join_node)
    graph.add_node("security_chunk_node", security_chunk_node)
    graph.add_node("compression_node", compression_node)
    graph.add_node("reranker_node", reranker_node)
    graph.add_node("citation_node", citation_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("reflection_node", reflection_node)
    graph.add_node("verification_node", verification_node)
    graph.add_node("formatter_node", formatter_node)
    graph.add_node("hitl_check_node", hitl_check_node)
    graph.add_node("hitl_pause_node", hitl_pause_node)
    graph.add_node("observability_node", observability_node)

    # ── Conditional Routing Functions ───────────────────────────────────────────
    def _route_after_security(state: MissionState) -> str:
        if not state.get("is_safe", True):
            return "security_block_node"
        return "planner_node"

    def _route_after_cache(state: MissionState) -> Any:
        if state.get("cached_report") is not None:
            return "observability_node"
        return ["vector_retrieval_node", "bm25_retrieval_node", "graph_retrieval_node"]

    def _route_after_verification(state: MissionState) -> str:
        if state.get("needs_retry", False):
            return "rewrite_node"
        return "formatter_node"

    def _route_after_hitl_check(state: MissionState) -> str:
        hitl = state.get("hitl_request")
        if hitl is not None and hitl.status == HITLStatus.PENDING:
            return "hitl_pause_node"
        return "observability_node"

    # ── Edge Wiring ─────────────────────────────────────────────────────────────
    graph.add_edge(START, "security_inbound_node")
    graph.add_conditional_edges("security_inbound_node", _route_after_security)
    graph.add_edge("security_block_node", "observability_node")
    graph.add_edge("planner_node", "rewrite_node")
    graph.add_edge("rewrite_node", "cache_node")
    graph.add_conditional_edges("cache_node", _route_after_cache)

    # Parallel branches join at research_join_node
    graph.add_edge("vector_retrieval_node", "research_join_node")
    graph.add_edge("bm25_retrieval_node", "research_join_node")
    graph.add_edge("graph_retrieval_node", "research_join_node")
    graph.add_edge("research_join_node", "security_chunk_node")
    graph.add_edge("security_chunk_node", "compression_node")
    graph.add_edge("compression_node", "reranker_node")
    graph.add_edge("reranker_node", "citation_node")
    graph.add_edge("citation_node", "synthesis_node")
    graph.add_edge("synthesis_node", "reflection_node")
    graph.add_edge("reflection_node", "verification_node")
    graph.add_conditional_edges("verification_node", _route_after_verification)
    graph.add_edge("formatter_node", "hitl_check_node")
    graph.add_conditional_edges("hitl_check_node", _route_after_hitl_check)
    graph.add_edge("hitl_pause_node", "observability_node")
    graph.add_edge("observability_node", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["hitl_pause_node"])
