import unittest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.data import ScoredDoc
from app.retrieval import (
    build_qdrant_filter,
    retrieve_vector,
    retrieve_bm25,
    merge_results,
)


class DummyBM25:
    def get_scores(self, tokens):
        # Return mock scores for 3 chunks
        return [2.5, 0.1, 5.0]


class TestRetrievalFunnel(unittest.TestCase):
    def test_build_qdrant_filter(self):
        filter_obj = build_qdrant_filter({"doc_type": "note", "is_table": True})
        self.assertIsNotNone(filter_obj)
        self.assertEqual(len(filter_obj.must), 2)

    def test_retrieve_bm25_with_metadata_filter(self):
        chunks = [
            Document(page_content="Chunk 1", metadata={"doc_type": "note", "source": "a.md"}),
            Document(page_content="Chunk 2", metadata={"doc_type": "external", "source": "b.pdf"}),
            Document(page_content="Chunk 3", metadata={"doc_type": "note", "source": "c.md"}),
        ]
        bm25 = DummyBM25()
        normalize_scores = lambda scores: [1.0 / (60 + i) for i in range(1, len(scores) + 1)]

        # Without filter
        results_all = retrieve_bm25(bm25, chunks, normalize_scores, "test query", k=10)
        self.assertEqual(len(results_all), 3)

        # With filter for doc_type = note (should exclude Chunk 2)
        results_filtered = retrieve_bm25(
            bm25, chunks, normalize_scores, "test query", k=10, filter_dict={"doc_type": "note"}
        )
        self.assertEqual(len(results_filtered), 2)
        sources = [r.source for r in results_filtered]
        self.assertIn("a.md", sources)
        self.assertIn("c.md", sources)
        self.assertNotIn("b.pdf", sources)

    def test_merge_results_rrf_ordering(self):
        doc1 = ScoredDoc(content="A", doc=None, vector_score=1.0 / 61, bm25_score=1.0 / 62, doc_id="id1")
        doc2 = ScoredDoc(content="B", doc=None, vector_score=1.0 / 65, bm25_score=0.0, doc_id="id2")
        weights = {"vector": 1.0, "bm25": 1.0, "feedback": 0.0}

        merged = merge_results([doc1, doc2], [], weights)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].doc_id, "id1")
        self.assertGreater(merged[0].hybrid_score, merged[1].hybrid_score)


if __name__ == "__main__":
    unittest.main()
