import concurrent.futures
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from app import config


@dataclass
class RAGReportCard:
    faithfulness: float
    context_relevance: float
    answer_relevance: float
    overall_grade: float
    latency_ms: float
    token_counts: Dict[str, int]
    production_cost_estimates_usd: Dict[str, float]
    flagged_hallucination: bool = False


class ContinuousEvaluator:
    """Continuous Evaluation & Monitoring Engine ('Continuous Evaluator (Heuristic Report Card)') with Split-Engine architecture.
    Synchronous per-query evaluation uses fast heuristic metrics (N-gram Jaccard entailment, Sigmoid cross-encoder relevance, and token overlap)."""

    def __init__(self, history_path: Optional[str] = None):
        self.history_path = history_path or getattr(
            config, "EVAL_HISTORY_PATH", "data/rag_eval_history.json"
        )
        self._async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _tokenize(self, text: str) -> set:
        words = re.findall(r"\w+", text.lower())
        return set(words)

    def grade_faithfulness(self, answer: str, retrieved_chunks: List[Any]) -> float:
        """Heuristic Faithfulness via N-gram Entailment Formula."""
        if not answer or not retrieved_chunks:
            return 0.0

        # Split answer into sentence assertions A
        raw_assertions = [s.strip() for s in re.split(r"[\.\!\?\n]+", answer) if len(s.strip()) > 10]
        if not raw_assertions:
            raw_assertions = [answer.strip()]

        # Extract text contents of retrieved chunks C
        chunk_texts = []
        for c in retrieved_chunks:
            t = getattr(c, "content", "") or getattr(c, "page_content", "") or str(c)
            chunk_texts.append(t)

        chunk_tokens_list = [self._tokenize(ct) for ct in chunk_texts]
        all_chunk_nums = set()
        for ct in chunk_texts:
            all_chunk_nums.update(re.findall(r"\b\d+(?:\.\d+)?\b", ct))

        assertion_scores = []
        for assertion in raw_assertions:
            a_tokens = self._tokenize(assertion)
            if not a_tokens:
                continue

            max_jaccard = 0.0
            for c_tokens in chunk_tokens_list:
                union = len(a_tokens | c_tokens)
                if union > 0:
                    jaccard = len(a_tokens & c_tokens) / union
                    if jaccard > max_jaccard:
                        max_jaccard = jaccard

            # Penalty check for explicit numeric/date tokens completely missing from context
            a_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", assertion))
            unsupported_nums = a_nums - all_chunk_nums
            if unsupported_nums and len(unsupported_nums) > 0:
                max_jaccard *= 0.15

            assertion_scores.append(max_jaccard)

        if not assertion_scores:
            return 0.0

        s_faith = sum(assertion_scores) / len(assertion_scores)
        return round(min(1.0, max(0.0, s_faith * 2.2)), 4)  # Scaled Jaccard

    def grade_context_relevance(self, retrieved_chunks: List[Any]) -> float:
        """Context Relevance via Sigmoid Cross-Encoder Reranker score normalization."""
        if not retrieved_chunks:
            return 0.0

        scores = []
        for c in retrieved_chunks:
            score = getattr(c, "rerank_score", None)
            if score is None:
                score = getattr(c, "hybrid_score", 0.0)
            scores.append(float(score))

        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)
        # Standard sigmoid mapping: 1 / (1 + exp(-x))
        try:
            s_relevance = 1.0 / (1.0 + math.exp(-avg_score))
        except OverflowError:
            s_relevance = 0.0 if avg_score < 0 else 1.0

        return round(s_relevance, 4)

    def grade_answer_relevance(self, query: str, answer: str) -> float:
        """Answer relevance grading query-to-answer token alignment."""
        q_tokens = self._tokenize(query)
        a_tokens = self._tokenize(answer)
        if not q_tokens or not a_tokens:
            return 0.0

        overlap = len(q_tokens & a_tokens)
        s_ans = overlap / max(1, len(q_tokens))
        return round(min(1.0, s_ans * 1.5), 4)

    def estimate_production_costs(
        self, prompt_tokens: int, completion_tokens: int
    ) -> Dict[str, float]:
        """Estimate exact cost across production cloud API models vs local compute."""
        pricing_dict = getattr(config, "PRODUCTION_PRICING_USD_PER_M_TOKENS", {})
        estimates = {}
        for model_name, rates in pricing_dict.items():
            in_cost = (prompt_tokens / 1_000_000.0) * rates.get("input", 0.0)
            out_cost = (completion_tokens / 1_000_000.0) * rates.get("output", 0.0)
            estimates[model_name] = round(in_cost + out_cost, 6)
        return estimates

    def grade_response(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Any],
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> RAGReportCard:
        """Synchronously compute triad heuristics (<2ms) and asynchronously log anomalous/sampled records."""
        tel = telemetry or {}
        lat_ms = float(tel.get("total_ms", 0.0))

        s_faith = self.grade_faithfulness(answer, retrieved_chunks)
        s_relevance = self.grade_context_relevance(retrieved_chunks)
        s_ans_rel = self.grade_answer_relevance(query, answer)

        overall = round((s_faith * 0.45) + (s_relevance * 0.35) + (s_ans_rel * 0.20), 4)
        flagged_hallucination = s_faith < 0.40

        # Estimate tokens roughly (~4 chars/token)
        p_tokens = max(10, len(query) // 4 + sum(len(getattr(c, "content", "")) // 4 for c in retrieved_chunks))
        c_tokens = max(5, len(answer) // 4)
        token_counts = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
        }

        prod_costs = self.estimate_production_costs(p_tokens, c_tokens)

        report_card = RAGReportCard(
            faithfulness=s_faith,
            context_relevance=s_relevance,
            answer_relevance=s_ans_rel,
            overall_grade=overall,
            latency_ms=lat_ms,
            token_counts=token_counts,
            production_cost_estimates_usd=prod_costs,
            flagged_hallucination=flagged_hallucination,
        )

        # Split-Engine async logging
        self._async_executor.submit(self._log_history_async, query, report_card)

        return report_card

    def _log_history_async(self, query: str, card: RAGReportCard):
        try:
            entry = {
                "timestamp": time.time(),
                "query_snippet": query[:80],
                "report_card": asdict(card),
            }
            _jsonl_logger_singleton(self.history_path).append(entry)
        except Exception:
            pass


def _jsonl_logger_singleton(path: str) -> "JSONLEvalLogger":
    """Return (or create) the process-wide JSONLEvalLogger for the given path."""
    if not hasattr(_jsonl_logger_singleton, "_instances"):
        _jsonl_logger_singleton._instances = {}
    if path not in _jsonl_logger_singleton._instances:
        _jsonl_logger_singleton._instances[path] = JSONLEvalLogger(path)
    return _jsonl_logger_singleton._instances[path]


class JSONLEvalLogger:
    """Append-only JSONL evaluation logger with size-based rotation.

    Rotation policy (fixed per architecture review):
      - Rotate when active file exceeds max_bytes (default 10 MB).
      - Rotated files renamed  rag_eval_history.YYYYMMDD-HHMMSS.jsonl.
      - Retain the max_rotations (default 5) most recent rotated files.
      - Older files are gzip-compressed (.jsonl.gz) rather than deleted.
    """

    _DEFAULT_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
    _DEFAULT_MAX_ROTATIONS = 5

    def __init__(
        self,
        log_path: str,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_rotations: int = _DEFAULT_MAX_ROTATIONS,
    ):
        self.log_path = log_path
        self.max_bytes = max_bytes
        self.max_rotations = max_rotations
        import threading
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        """Thread-safely append one JSON record as a JSONL line, rotating if needed."""
        import gzip as _gzip
        import shutil
        from datetime import datetime

        with self._lock:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

            # Rotate if active file has grown past threshold
            if (
                os.path.exists(self.log_path)
                and os.path.getsize(self.log_path) >= self.max_bytes
            ):
                self._rotate(_gzip, shutil, datetime)

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def _rotate(self, _gzip, shutil, datetime) -> None:
        """Rename active file, compress overflow rotations, prune old files."""
        import glob

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = os.path.splitext(self.log_path)[0]   # strip .jsonl
        rotated_name = f"{stem}.{ts}.jsonl"
        os.rename(self.log_path, rotated_name)

        # Collect all rotated files (plain .jsonl and .jsonl.gz), sort oldest-first
        pattern_plain = f"{stem}.????????-??????.jsonl"
        pattern_gz    = f"{stem}.????????-??????.jsonl.gz"
        plain = sorted(glob.glob(pattern_plain))
        gzipped = sorted(glob.glob(pattern_gz))
        all_rotated = sorted(plain + gzipped)

        # Keep the max_rotations most recent as plain; compress the rest
        keep_plain = set(plain[-self.max_rotations:]) if plain else set()
        for path in plain:
            if path not in keep_plain:
                gz_path = path + ".gz"
                with open(path, "rb") as f_in, _gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)

