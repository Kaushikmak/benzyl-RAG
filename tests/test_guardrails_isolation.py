import unittest
from guardrails.core.base import Candidate
from guardrails.defenses.isolation import PartitionAggregateIsolation


class TestGuardrailsIsolation(unittest.TestCase):
    def test_partition_and_aggregate(self):
        c1 = Candidate(doc_id="c1", content="Fact A", source="n1.md")
        c2 = Candidate(doc_id="c2", content="Fact B", source="n2.md")
        c3 = Candidate(doc_id="c3", content="Fact C", source="n3.md")
        c4 = Candidate(doc_id="c4", content="Fact D", source="n4.md")

        iso = PartitionAggregateIsolation(num_partitions=2)
        parts = iso.partition_candidates([c1, c2, c3, c4])

        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0]), 2)
        self.assertEqual(len(parts[1]), 2)

        def sub_gen(query, cands):
            return " + ".join(c.content for c in cands)

        def agg(query, answers):
            return " | ".join(answers)

        final_ans = iso.generate_isolated_answer("test", [c1, c2, c3, c4], sub_gen, agg)
        self.assertEqual(final_ans, "Fact A + Fact C | Fact B + Fact D")


if __name__ == "__main__":
    unittest.main()
