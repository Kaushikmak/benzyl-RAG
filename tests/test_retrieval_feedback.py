import sqlite3
import unittest

from app.feedback import compute_feedback_priors, init_feedback_tables, save_answer_record, save_feedback
from app.retrieval import merge_results
from app.data import ScoredDoc


class TestRetrievalFeedback(unittest.TestCase):
    def test_weighted_fusion_prefers_combined_signal(self):
        d1 = ScoredDoc(content="a", doc=None, doc_id="1", vector_score=0.9, bm25_score=0.1)
        d2 = ScoredDoc(content="b", doc=None, doc_id="2", vector_score=0.2, bm25_score=0.8)
        merged = merge_results([d1], [d2], {"vector": 1.0, "bm25": 1.0, "feedback": 0.0}, {})
        self.assertIn(merged[0].doc_id, {"1", "2"})
        self.assertEqual(len(merged), 2)

    def test_feedback_priors_require_min_votes_and_are_bounded(self):
        conn = sqlite3.connect(":memory:")
        init_feedback_tables(conn)

        save_answer_record(conn, "a1", "s1", "q", "ans", ["chunk-1"], ["n1.md"], [])
        save_feedback(conn, "a1", "s1", "up", [], "")
        priors = compute_feedback_priors(conn)
        self.assertNotIn("chunk-1", priors)

        save_feedback(conn, "a1", "s1", "up", [], "")
        priors = compute_feedback_priors(conn)
        self.assertIn("chunk-1", priors)
        self.assertTrue(-0.25 <= priors["chunk-1"] <= 0.25)


if __name__ == "__main__":
    unittest.main()
