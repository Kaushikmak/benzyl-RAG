import logging
from rank_bm25 import BM25Okapi
from indexing.tokenizer import tokenize

logger = logging.getLogger(__name__)

def build_bm25(chunks):
    logger.info("Creating BM25 index...")
    chunk_texts = [chunk.page_content for chunk in chunks]
    tokenized_chunks = [tokenize(text) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    logger.info("BM25 index created")

    return bm25