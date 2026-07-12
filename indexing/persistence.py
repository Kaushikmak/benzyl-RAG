import pickle
import logging
import app.config as config
import os

logger = logging.getLogger(__name__)

def save_graph(graph):
    os.makedirs(os.path.dirname(config.GRAPH_PATH), exist_ok=True)
    logger.info("Saving graph index")
    with open(config.GRAPH_PATH,"wb") as f:
        pickle.dump(graph, f)
    logger.info("Graph saved")

def save_bm25(bm25,chunks):
    logger.info("Saving BM25 index")
    with open(config.BM25_PATH,"wb") as f:
        pickle.dump((bm25, chunks),f)
    logger.info("BM25 saved")