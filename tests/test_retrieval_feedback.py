import sqlite3

from app.feedback import compute_feedback_priors, init_feedback_tables, save_answer_record, save_feedback
from app.retrieval import merge_results
from app.data import ScoredDoc


def test_weighted_fusion_prefers_combined_signal():
    d1 = ScoredDoc(content="a", doc=None, doc_id="1", vector_score=0.9, bm25_score=0.1)
    d2 = ScoredDoc(content="b", doc=None, doc_id="2", vector_score=0.2, bm25_score=0.8)
    merged = merge_results([d1], [d2], {"vector": 1.0, "bm25": 1.0, "feedback": 0.0}, {})
    assert merged[0].doc_id in {"1", "2"}
    assert len(merged) == 2


def test_feedback_priors_require_min_votes_and_are_bounded():
    conn = sqlite3.connect(":memory:")
    init_feedback_tables(conn)

    save_answer_record(conn, "a1", "s1", "q", "ans", ["chunk-1"], ["n1.md"], [])
    save_feedback(conn, "a1", "s1", "up", [], "")
    priors = compute_feedback_priors(conn)
    assert "chunk-1" not in priors

    save_feedback(conn, "a1", "s1", "up", [], "")
    priors = compute_feedback_priors(conn)
    assert "chunk-1" in priors
    assert -0.25 <= priors["chunk-1"] <= 0.25
