import os
import shutil
import tempfile
import unittest

from guardrails.evaluation.safety_net import snapshot_index_to_clone, get_clone_config, clean_clone


class TestGuardrailsSafetyNet(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.src_vector = os.path.join(self.test_dir, "vectorstore")
        self.src_node = os.path.join(self.test_dir, "node_vectorstore")
        self.src_bm25 = os.path.join(self.test_dir, "bm25.pkl")
        self.src_graph = os.path.join(self.test_dir, "graph.pkl")
        self.clone_dir = os.path.join(self.test_dir, "clone")

        os.makedirs(self.src_vector, exist_ok=True)
        os.makedirs(self.src_node, exist_ok=True)
        with open(os.path.join(self.src_vector, "test.txt"), "w") as f:
            f.write("vector data")
        with open(self.src_bm25, "w") as f:
            f.write("bm25 data")
        with open(self.src_graph, "w") as f:
            f.write("graph data")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_snapshot_and_clean_clone(self):
        paths = snapshot_index_to_clone(
            source_vectorstore=self.src_vector,
            source_node_vectorstore=self.src_node,
            source_bm25=self.src_bm25,
            source_graph=self.src_graph,
            clone_dir=self.clone_dir,
        )

        self.assertTrue(os.path.exists(paths["VECTORSTORE_DIR"]))
        self.assertTrue(os.path.exists(paths["BM25_PATH"]))
        self.assertTrue(os.path.exists(paths["GRAPH_PATH"]))

        config_override = get_clone_config(self.clone_dir)
        self.assertEqual(config_override["BM25_PATH"], paths["BM25_PATH"])

        clean_clone(self.clone_dir)
        self.assertFalse(os.path.exists(self.clone_dir))


if __name__ == "__main__":
    unittest.main()
