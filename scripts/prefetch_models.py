import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prefetch_models")

def prefetch():
    logger.info("Starting build-time prefetch of HuggingFace models...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("Downloading embedding model: BAAI/bge-m3...")
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model BAAI/bge-m3 downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to prefetch embedding model: {e}")
        sys.exit(1)

    try:
        from sentence_transformers import CrossEncoder
        logger.info("Downloading reranker model: BAAI/bge-reranker-v2-m3...")
        CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
        logger.info("Reranker model BAAI/bge-reranker-v2-m3 downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to prefetch reranker model: {e}")
        sys.exit(1)

    logger.info("All models prefetched successfully into /root/.cache/huggingface")

if __name__ == "__main__":
    prefetch()
