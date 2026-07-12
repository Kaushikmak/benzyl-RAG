import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass

from app.evaluation import ContinuousEvaluator


@dataclass
class DummyCandidate:
    content: str
    rerank_score: float = 2.0


class TestContinuousEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.history_path = os.path.join(self.test_dir, "eval_history.json")
        self.evaluator = ContinuousEvaluator(history_path=self.history_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_faithfulness_grounded_vs_hallucinated(self):
        chunk = DummyCandidate(content="Universal RAG indexed 150 documents in 2024.")
        # Grounded answer
        s_grounded = self.evaluator.grade_faithfulness(
            "Universal RAG indexed 150 documents in 2024.", [chunk]
        )
        self.assertGreaterEqual(s_grounded, 0.70)

        # Hallucinated numeric assertion
        s_hallucinated = self.evaluator.grade_faithfulness(
            "Universal RAG indexed 9999 documents in 1999.", [chunk]
        )
        self.assertLess(s_hallucinated, 0.40)

    def test_sigmoid_context_relevance(self):
        high_chunk = DummyCandidate(content="Relevant text", rerank_score=3.0)
        s_rel = self.evaluator.grade_context_relevance([high_chunk])
        self.assertGreater(s_rel, 0.90)  # sigmoid(3.0) ~ 0.9526

        low_chunk = DummyCandidate(content="Irrelevant text", rerank_score=-2.0)
        s_low = self.evaluator.grade_context_relevance([low_chunk])
        self.assertLess(s_low, 0.20)  # sigmoid(-2.0) ~ 0.1192

    def test_production_cost_estimation(self):
        costs = self.evaluator.estimate_production_costs(
            prompt_tokens=1000, completion_tokens=500
        )
        self.assertIn("gpt-4o", costs)
        self.assertIn("claude-3-5-sonnet", costs)
        self.assertIn("local-compute", costs)
        self.assertEqual(costs["local-compute"], 0.0)
        self.assertGreater(costs["gpt-4o"], 0.0)

    def test_grade_response_report_card(self):
        chunk = DummyCandidate(content="Local Vector DB search returned 50 items.")
        card = self.evaluator.grade_response(
            query="How many items returned?",
            answer="Vector DB search returned 50 items.",
            retrieved_chunks=[chunk],
            telemetry={"total_ms": 12.5},
        )
        self.assertFalse(card.flagged_hallucination)
        self.assertEqual(card.latency_ms, 12.5)
        self.assertGreater(card.overall_grade, 0.50)


if __name__ == "__main__":
    unittest.main()
