import unittest
from app.router import QueryRouter, eval_math_ast


class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()
        self.sys_ctx = {
            "notes_count": 142,
            "chunks_count": 890,
            "guardrails_active": True,
            "ragshield_active": True,
        }

    def test_eval_math_ast_valid(self):
        self.assertEqual(eval_math_ast("145 * 2 + 10"), 300.0)
        self.assertAlmostEqual(eval_math_ast("sqrt(144) + pow(2, 3)"), 20.0)

    def test_eval_math_ast_security_rejection(self):
        with self.assertRaises(ValueError):
            eval_math_ast("__import__('os').system('ls')")

    def test_direct_math_route(self):
        decision = self.router.triage("145 * 23 + sqrt(144)", self.sys_ctx)
        self.assertEqual(decision.action, "DIRECT_MATH")
        self.assertIn("3347", decision.direct_answer)

    def test_semantic_math_trap_prevention(self):
        query = "What was our total revenue growth if Q1 was 4.5M and Q2 was 5.1M?"
        decision = self.router.triage(query, self.sys_ctx)
        self.assertEqual(decision.action, "DOCUMENT_RAG")

    def test_direct_status_route(self):
        decision = self.router.triage("how many documents are indexed?", self.sys_ctx)
        self.assertEqual(decision.action, "DIRECT_STATUS")
        self.assertIn("142", decision.direct_answer)
        self.assertIn("890", decision.direct_answer)

    def test_direct_conversation_route(self):
        decision = self.router.triage("hello", self.sys_ctx)
        self.assertEqual(decision.action, "DIRECT_CONVERSATION")

    def test_document_rag_default_route(self):
        decision = self.router.triage("How does PGVector store dense embeddings?", self.sys_ctx)
        self.assertEqual(decision.action, "DOCUMENT_RAG")


if __name__ == "__main__":
    unittest.main()
