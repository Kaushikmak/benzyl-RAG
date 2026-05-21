import os
import shutil
import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import app.config as config

logger = logging.getLogger(__name__)

def load_embedding_model():
    logger.info("Loading embedding model")
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL,model_kwargs={"device": config.DEVICE})
    logger.info("Embedding model loaded")
    return embedding_model


def build_vectorstores(chunks,graph,embedding_model):
    if os.path.exists(config.VECTORSTORE_DIR):
        shutil.rmtree(config.VECTORSTORE_DIR)

    if os.path.exists(config.NODE_VECTORSTORE_DIR):
        shutil.rmtree(config.NODE_VECTORSTORE_DIR)

    logger.info("Building chunk vector database")
    Chroma.from_documents(chunks,embedding_model,persist_directory=config.VECTORSTORE_DIR)
    logger.info("Chunk vector database created")

    logger.info("Building node vector database")
    node_names = list(graph.nodes)
    Chroma.from_texts(
        texts=node_names,
        embedding=embedding_model,
        persist_directory=config.NODE_VECTORSTORE_DIR
    )
    logger.info("Node vector database created")