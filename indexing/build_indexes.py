import logging
from indexing.document_loader import load_documents
from indexing.graph_builder import build_graph
from indexing.chunking import chunk_documents
from indexing.bm25_builder import build_bm25
from indexing.vector_builder import load_embedding_model,build_vectorstores
from indexing.persistence import save_graph, save_bm25


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Indexing started")
    documents = load_documents()
    graph = build_graph(documents)
    chunks = chunk_documents(documents)
    bm25 = build_bm25(chunks)
    save_graph(graph)
    save_bm25(bm25,chunks)
    embedding_model = (load_embedding_model())
    build_vectorstores(chunks,graph,embedding_model)
    logger.info("Indexing complete")

if __name__ == "__main__":
    main()