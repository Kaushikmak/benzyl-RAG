import logging
import pickle
import numpy as np
import ollama
import os
import re
from typing import List, Dict
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from app import config
from app.retrieval import (retrieve_vector,retrieve_bm25,merge_results)
from app.data import ScoredDoc
from app.graph_search import expand_with_graph
from app.reranker import rerank



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class ImprovedRAG:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={
                "device": config.DEVICE
            }
        )

        logger.info("Loading vector database...")

        self.db = Chroma(
            persist_directory=config.VECTORSTORE_DIR,
            embedding_function=self.embedding_model
        )

        logger.info("Loading node vector database...")

        self.node_db = Chroma(
            persist_directory=config.NODE_VECTORSTORE_DIR,
            embedding_function=self.embedding_model
        )

        logger.info("Loading BM25 index...")

        with open(config.BM25_PATH, "rb") as f:
            self.bm25, self.bm25_chunks = pickle.load(f)

        logger.info("Loading graph...")

        with open(config.GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)

        logger.info("Building chunk index...")

        self.chunk_index = self._build_chunk_index()

        logger.info("Loading reranker...")

        self.reranker = CrossEncoder(
            config.RERANK_MODEL,
            device=config.DEVICE
        )

        logger.info(
            f"Loaded {len(self.graph.nodes)} notes "
            f"and {len(self.bm25_chunks)} chunks"
        )

    def _build_chunk_index(self) -> Dict[str, List]:

        index = defaultdict(list)

        for chunk in self.bm25_chunks:

            source = chunk.metadata.get("source", "")

            filename = os.path.basename(source)

            note_name = filename.replace(".md", "")

            index[note_name].append(chunk)

        return index

    def _normalize_scores(self,scores: List[float],k: int = 60) -> List[float]:

        ranked_indices = np.argsort(scores)[::-1]

        normalized = np.zeros(len(scores))

        for rank, idx in enumerate(ranked_indices, start=1):

            normalized[idx] = 1.0 / (k + rank)

        return normalized.tolist()

    def build_context(self,docs: List[ScoredDoc]) -> str:

        parts = []

        for doc in docs:

            parts.append(
                f"""
                SOURCE: {doc.source}

                CONTENT:
                {doc.content}
                """
            )

        separator = "\n\n" + ("=" * 50) + "\n\n"

        return separator.join(parts)

    def answer(
        self,
        query: str,
        verbose: bool = False
    ) -> dict:

        # VECTOR SEARCH
        vector_docs = retrieve_vector(
            self.db,
            self._normalize_scores,
            query,
            k=6
        )

        # BM25 SEARCH
        bm25_docs = retrieve_bm25(
            self.bm25,
            self.bm25_chunks,
            self._normalize_scores,
            query,
            k=6
        )

        # HYBRID MERGE
        merged = merge_results(
            vector_docs,
            bm25_docs
        )

        if verbose:

            print("\nHYBRID RETRIEVAL\n")

            for i, doc in enumerate(merged[:5], start=1):

                print(
                    f"{i}. [{doc.source}] "
                    f"Hybrid Score: {doc.hybrid_score:.4f}"
                )

                print(f"   {doc.content[:120]}...\n")

        # GRAPH EXPANSION
        top_merged = merged[:4]

        expanded = expand_with_graph(
            self.graph,
            self.chunk_index,
            top_merged,
            max_neighbors=2,
            max_chunks_per_neighbor=2
        )

        if verbose:

            print("\nGRAPH EXPANSION\n")

            print(
                f"Expanded from "
                f"{len(top_merged)} -> {len(expanded)} chunks"
            )

        # RERANK
        reranked = rerank(
            self.reranker,
            query,
            expanded,
            top_k=3
        )

        if verbose:

            print("\nRERANKED RESULTS\n")

            for i, doc in enumerate(reranked, start=1):

                print(
                    f"{i}. [{doc.source}] "
                    f"Rerank Score: {doc.rerank_score:.4f}"
                )

                print(f"   {doc.content[:120]}...\n")

        # BUILD CONTEXT
        context = self.build_context(reranked)

        prompt = f"""
                You are an intelligent assistant reasoning over an Obsidian vault.

                Rules:
                1. Base your answer primarily on the provided context.
                2. You may use general programming knowledge to interpret the user's query and the context, especially for code snippets.
                3. Do NOT hallucinate information about the user's personal vault that is not in the context.
                4. If the context does not contain the answer and you cannot deduce it from the provided code and context, reply:
                "I could not find this information in the Obsidian vault."

                Context:
                {context}

                Question:
                {query}

                Answer:
                """

        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            options={
                "temperature": 0
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "answer": response["message"]["content"],
            "sources": list(set([doc.source for doc in reranked]))
        }

    def show_neighbors(self,note_query: str,depth: int = 1):
        node_map_lower = {
            node.lower(): node
            for node in self.graph.nodes()
        }

        results = self.node_db.similarity_search(note_query,k=3)

        if not results:
            print("\nNo matching notes found.")
            return

        print("\nSemantic Matches:\n")
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result.page_content}")

        matched_node = results[0].page_content
        actual_node = node_map_lower.get(matched_node.lower())

        if not actual_node:
            print(
                f"Error: '{matched_node}' "
                f"not found in graph"
            )

            return

        print(f"\nUsing: {actual_node}")

        visited = set()

        current = {actual_node}

        for d in range(depth):
            next_nodes = set()
            print(f"\nDepth {d + 1}:\n")
            for node in current:
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        print(f"{node} -> {neighbor}")
                        next_nodes.add(neighbor)
                        visited.add(neighbor)

            current = next_nodes


def main():

    rag = ImprovedRAG()

    print("\nCommands:")
    print("  /verbose on")
    print("  /verbose off")
    print("  /graph[N] <note>")
    print("  exit | quit | q | /bye")

    verbose = False

    while True:

        query = input("\nAsk: ").strip()

        # EXIT
        if query.lower() in [
            "exit",
            "quit",
            "q",
            "/bye"
        ]:
            break

        # VERBOSE
        if query.lower() == "/verbose on":

            verbose = True

            print("Verbose mode enabled")

            continue

        elif query.lower() == "/verbose off":

            verbose = False

            print("Verbose mode disabled")

            continue

        # GRAPH COMMAND
        graph_match = re.match(
            r"/graph(\d+)?\s+(.+)",
            query
        )

        if graph_match:

            depth = (
                int(graph_match.group(1))
                if graph_match.group(1)
                else 1
            )

            note = graph_match.group(2).strip()

            rag.show_neighbors(
                note,
                depth=depth
            )

            continue

        # NORMAL QA
        result = rag.answer(
            query,
            verbose=verbose
        )

        print("\nAnswer:\n")

        print(result["answer"])


if __name__ == "__main__":
    main()