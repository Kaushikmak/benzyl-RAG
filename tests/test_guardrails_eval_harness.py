import os
import shutil
import tempfile
import unittest

from guardrails.redteaming.corpus_poisoning import generate_poisoned_chunk, generate_near_duplicate_cluster
from guardrails.redteaming.indirect_injection import apply_injection_payload
from guardrails.core.base import Candidate
from guardrails.evaluation.harness import EvalHarness, compute_asr, compute_ndcg


class TestGuardrailsEvalHarness(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.results_path = os.path.join(self.test_dir, "eval_results.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_attack_generation_and_harness(self):
        clean_c1 = Candidate(doc_id="c1", content="Clean content 1", source="note1.md")
        clean_c2 = Candidate(doc_id="c2", content="Clean content 2", source="note2.md")

        poisoned = generate_poisoned_chunk("password", "EXFILTRATION_PAYLOAD")
        injected = apply_injection_payload(clean_c2, "IGNORE_PREVIOUS")

        retrieved_clean = [clean_c1, clean_c2]
        retrieved_poisoned = [poisoned, injected, clean_c1]

        asr = compute_asr(retrieved_poisoned)
        self.assertAlmostEqual(asr, 2.0 / 3.0, places=3)

        ndcg = compute_ndcg(["c1", "c2"], ["c1", "c2"])
        self.assertAlmostEqual(ndcg, 1.0, places=3)

        harness = EvalHarness(self.results_path)
        harness.evaluate_run(
            config_name="Baseline (Poisoned)",
            retrieved_clean=retrieved_clean,
            retrieved_poisoned=retrieved_poisoned,
            target_relevant_ids=["c1", "c2"],
            latency_ms=12.5,
        )

        table = harness.summary_table()
        self.assertIn("Baseline (Poisoned)", table)
        self.assertIn("66.7%", table)


if __name__ == "__main__":
    unittest.main()
