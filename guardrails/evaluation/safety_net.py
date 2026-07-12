"""Safety net to snapshot and isolate RAG indexes for evaluation and attack generation."""

import logging
import os
import shutil
from typing import Dict

logger = logging.getLogger(__name__)

CLONE_DIR = "./data/clone"


def snapshot_index_to_clone(
    source_vectorstore: str = "./data/qdrant_db",
    source_node_vectorstore: str = "./data/qdrant_db",
    source_bm25: str = "./data/bm25.pkl",
    source_graph: str = "./data/graph.pkl",
    clone_dir: str = CLONE_DIR,
) -> Dict[str, str]:
    """Copy the existing live index artifacts into a throwaway clone directory."""
    os.makedirs(clone_dir, exist_ok=True)

    clone_vectorstore = os.path.join(clone_dir, "vectorstore")
    clone_node_vectorstore = os.path.join(clone_dir, "node_vectorstore")
    clone_bm25 = os.path.join(clone_dir, "bm25.pkl")
    clone_graph = os.path.join(clone_dir, "graph.pkl")

    # Snapshot vectorstore
    if os.path.exists(source_vectorstore):
        if os.path.exists(clone_vectorstore):
            shutil.rmtree(clone_vectorstore)
        shutil.copytree(source_vectorstore, clone_vectorstore)
        logger.info(f"Snapshotted vectorstore to {clone_vectorstore}")

    # Snapshot node_vectorstore
    if os.path.exists(source_node_vectorstore):
        if os.path.exists(clone_node_vectorstore):
            shutil.rmtree(clone_node_vectorstore)
        shutil.copytree(source_node_vectorstore, clone_node_vectorstore)
        logger.info(f"Snapshotted node_vectorstore to {clone_node_vectorstore}")

    # Snapshot BM25
    if os.path.exists(source_bm25):
        shutil.copy2(source_bm25, clone_bm25)
        logger.info(f"Snapshotted BM25 index to {clone_bm25}")

    # Snapshot Graph
    if os.path.exists(source_graph):
        shutil.copy2(source_graph, clone_graph)
        logger.info(f"Snapshotted Graph index to {clone_graph}")

    return {
        "VECTORSTORE_DIR": clone_vectorstore,
        "NODE_VECTORSTORE_DIR": clone_node_vectorstore,
        "BM25_PATH": clone_bm25,
        "GRAPH_PATH": clone_graph,
    }


def get_clone_config(clone_dir: str = CLONE_DIR) -> Dict[str, str]:
    """Return dictionary of path overrides pointing to clone index."""
    return {
        "VECTORSTORE_DIR": os.path.join(clone_dir, "vectorstore"),
        "NODE_VECTORSTORE_DIR": os.path.join(clone_dir, "node_vectorstore"),
        "BM25_PATH": os.path.join(clone_dir, "bm25.pkl"),
        "GRAPH_PATH": os.path.join(clone_dir, "graph.pkl"),
    }


def clean_clone(clone_dir: str = CLONE_DIR):
    """Remove throwaway clone directory."""
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
        logger.info(f"Cleaned clone directory {clone_dir}")
