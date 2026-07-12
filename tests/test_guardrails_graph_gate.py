import unittest
from guardrails.core.base import Candidate
from guardrails.defenses.detection.graph_gate import GraphExpansionGate


class TestGuardrailsGraphGate(unittest.TestCase):
    def test_graph_gate_filters_low_similarity_neighbors(self):
        c1 = Candidate(doc_id="good_neighbor", content="Legitimate neighbor", source="n1.md", scores={"vector": 0.45})
        c2 = Candidate(doc_id="poison_neighbor", content="Poisoned link neighbor", source="n2.md", scores={"vector": 0.08})

        gate = GraphExpansionGate(threshold=0.25)
        filtered = gate.filter_candidates([c1, c2], query="test query")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].doc_id, "good_neighbor")

        sweep = gate.sweep_thresholds([c1, c2], query="test query", thresholds=[0.05, 0.25, 0.50])
        self.assertEqual(sweep[0.05]["admitted_count"], 2)
        self.assertEqual(sweep[0.25]["admitted_count"], 1)
        self.assertEqual(sweep[0.50]["admitted_count"], 0)


if __name__ == "__main__":
    unittest.main()
