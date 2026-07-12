import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import shutil
import logging
import torch
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import app.config as config

logger = logging.getLogger(__name__)

def load_embedding_model():
    logger.info("Loading embedding model")
    batch_size = getattr(config, "EMBED_BATCH_SIZE", 4)
    device = config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        )
    except Exception as e:
        logger.warning(f"Failed to load embedding model on {device}: {e}. Trying CPU fallback.")
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        )
    logger.info("Embedding model loaded")
    return embedding_model


def build_vectorstores(chunks, graph, embedding_model):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if os.path.exists(config.QDRANT_DIR):
        shutil.rmtree(config.QDRANT_DIR)
    os.makedirs(config.QDRANT_DIR, exist_ok=True)

    batch_size = getattr(config, "EMBED_BATCH_SIZE", 8)

    logger.info("Building chunk Qdrant vector database")
    QdrantVectorStore.from_documents(
        chunks,
        embedding=embedding_model,
        path=config.QDRANT_DIR,
        collection_name=config.QDRANT_CHUNK_COLLECTION,
        force_recreate=True,
        batch_size=batch_size,
    )
    logger.info("Chunk Qdrant vector database created")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Building node Qdrant vector database")
    node_names = list(graph.nodes)
    QdrantVectorStore.from_texts(
        texts=node_names,
        embedding=embedding_model,
        path=config.QDRANT_DIR,
        collection_name=config.QDRANT_NODE_COLLECTION,
        force_recreate=True,
        batch_size=batch_size,
    )
    logger.info("Node Qdrant vector database created")