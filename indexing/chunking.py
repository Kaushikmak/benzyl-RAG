import os
import logging
from collections import Counter
from langchain_text_splitters import (RecursiveCharacterTextSplitter)
from app import config

logger = logging.getLogger(__name__)

def chunk_documents(documents):
    logger.info("Chunking documents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,chunk_overlap=config.CHUNK_OVERLAP,separators=["\n\n","\n",". "," ",""],length_function=len)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        source = chunk.metadata.get("source","")
        filename = os.path.basename(source)
        doc_type = chunk.metadata.get("doc_type", "note")
        if doc_type == "note":
            note_name = filename.replace(".md","")
        else:
            note_name = f"ext::{os.path.splitext(filename)[0]}"
        chunk.metadata["note_name"] = note_name
        chunk.metadata["filename"] = filename

    logger.info(f"Created {len(chunks)} chunks")
    chunks_per_note = Counter(chunk.metadata["note_name"] for chunk in chunks)
    logger.info("Chunk statistics:")
    logger.info(
        f"Min chunks/note: "
        f"{min(chunks_per_note.values())}"
    )
    logger.info(
        f"Max chunks/note: "
        f"{max(chunks_per_note.values())}"
    )
    logger.info(
        f"Average chunks/note: "
        f"{sum(chunks_per_note.values()) / len(chunks_per_note):.2f}"
    )
    return chunks
