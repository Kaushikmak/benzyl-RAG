import logging
from langchain_community.document_loaders import (DirectoryLoader,TextLoader)
import app.config as config

logger = logging.getLogger(__name__)

def load_documents():
    logger.info(f"Loading documents from {config.VAULT_PATH}")
    loader = DirectoryLoader(config.VAULT_PATH,glob="**/*.md",loader_cls=TextLoader,show_progress=True)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    return documents