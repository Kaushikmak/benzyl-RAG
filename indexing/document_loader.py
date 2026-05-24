import glob
import logging
import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader

import app.config as config

logger = logging.getLogger(__name__)


def _load_markdown_documents() -> List:
    loader = DirectoryLoader(
        config.VAULT_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["doc_type"] = "note"
    return docs


def _load_external_documents() -> List:
    if not os.path.isdir(config.EXTERNAL_DOCS_PATH):
        return []

    docs = []
    for ext in config.EXTERNAL_DOC_EXTENSIONS:
        pattern = os.path.join(config.EXTERNAL_DOCS_PATH, f"**/*{ext}")
        for file_path in glob.glob(pattern, recursive=True):
            try:
                loaded = UnstructuredFileLoader(file_path).load()
                for doc in loaded:
                    doc.metadata["doc_type"] = "external"
                    doc.metadata["external_ext"] = ext
                    doc.metadata["source"] = file_path
                docs.extend(loaded)
            except Exception as e:
                logger.warning("Failed to load %s: %s", file_path, e)
    return docs


def load_documents():
    logger.info("Loading documents from %s", config.VAULT_PATH)
    notes = _load_markdown_documents()
    external_docs = _load_external_documents()
    documents = notes + external_docs
    logger.info("Loaded %s notes and %s external docs", len(notes), len(external_docs))
    return documents
