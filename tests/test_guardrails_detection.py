import unittest
import numpy as np

from guardrails.core.base import Candidate
from guardrails.defenses.detection import EmbeddingOutlierDetector, NearDuplicateClusterDetector


class TestGuardrailsDetection(unittest.TestCase):
    def test_near_duplicate_cluster_detector(self):
        c1 = Candidate(doc_id="c1", content="Alpha Beta Gamma Delta Epsilon", source="n1.md")
        c2 = Candidate(doc_id="c2", content="Alpha Beta Gamma Delta Epsilon extra", source="n2.md")
        c3 = Candidate(doc_id="c3", content="Completely unique documents about system architecture", source="n3.md")

        detector = NearDuplicateClusterDetector(similarity_threshold=0.80, max_duplicates_allowed=1)
        filtered = detector.filter_candidates([c1, c2, c3], query="system")

        self.assertEqual(len(filtered), 2)
        self.assertTrue(c2.quarantined)

    def test_embedding_outlier_detector(self):
        c1 = Candidate(doc_id="c1", content="Note 1", source="n1.md")
        c2 = Candidate(doc_id="c2", content="Note 2", source="n2.md")
        c3 = Candidate(doc_id="c3", content="Note 3", source="n3.md")
        c4 = Candidate(doc_id="outlier", content="Outlier note", source="n4.md")

        embs = [
            np.array([1.0, 1.0]),
            np.array([1.01, 0.99]),
            np.array([0.99, 1.01]),
            np.array([50.0, -45.0]),  # Extreme outlier
        ]

        detector = EmbeddingOutlierDetector(max_z_score=1.5)
        filtered = detector.filter_candidates([c1, c2, c3, c4], query="query", embeddings=embs)

        self.assertEqual(len(filtered), 3)
        self.assertTrue(c4.quarantined)


if __name__ == "__main__":
    unittest.main()
