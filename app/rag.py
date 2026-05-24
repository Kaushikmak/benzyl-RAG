import logging
import os
import pickle
import re
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import ollama
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from app import config
from app.data import ScoredDoc
from app.graph_search import expand_with_graph
from app.reranker import rerank
from app.retrieval import merge_results, retrieve_bm25, retrieve_vector


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class ImprovedRAG:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": config.DEVICE},
        )

        self.db = Chroma(
            persist_directory=config.VECTORSTORE_DIR,
            embedding_function=self.embedding_model,
        )

        self.node_db = Chroma(
            persist_directory=config.NODE_VECTORSTORE_DIR,
            embedding_function=self.embedding_model,
        )

        with open(config.BM25_PATH, "rb") as f:
            self.bm25, self.bm25_chunks = pickle.load(f)

        with open(config.GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)

        self.chunk_index = self._build_chunk_index()

        self.reranker = CrossEncoder(config.RERANK_MODEL, device=config.DEVICE)

        self.query_cache: Dict[str, dict] = {}
        self.feedback_priors: Dict[str, float] = {}

        logger.info(
            "Loaded %s notes and %s chunks", len(self.graph.nodes), len(self.bm25_chunks)
        )

    def _build_chunk_index(self) -> Dict[str, List]:
        index = defaultdict(list)
        for chunk in self.bm25_chunks:
            source = chunk.metadata.get("source", "")
            filename = os.path.basename(source)
            note_name = filename.replace(".md", "")
            index[note_name].append(chunk)
        return index

    def _normalize_scores(self, scores: List[float], k: int = 60) -> List[float]:
        ranked_indices = np.argsort(scores)[::-1]
        normalized = np.zeros(len(scores))
        for rank, idx in enumerate(ranked_indices, start=1):
            normalized[idx] = 1.0 / (k + rank)
        return normalized.tolist()

    def _weights(self) -> Dict[str, float]:
        return {
            "vector": config.W_VECTOR,
            "bm25": config.W_BM25,
            "feedback": config.W_FEEDBACK,
        }

    def set_feedback_priors(self, priors: Dict[str, float]):
        self.feedback_priors = priors

    def build_context(self, docs: List[ScoredDoc]) -> str:
        parts = []
        for doc in docs:
            source_tag = f"[{doc.source_kind}]"
            parts.append(
                f"SOURCE {source_tag}: {doc.source}\nDOC_ID: {doc.doc_id}\n\nCONTENT:\n{doc.content}"
            )
        return "\n\n" + ("=" * 50) + "\n\n".join(parts)

    def _cache_get(self, key: str) -> Optional[dict]:
        if not config.ENABLE_QUERY_CACHE:
            return None
        return self.query_cache.get(key)

    def _cache_put(self, key: str, value: dict):
        if not config.ENABLE_QUERY_CACHE:
            return
        if len(self.query_cache) >= config.QUERY_CACHE_SIZE:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[key] = value

    def retrieve_candidates(self, query: str, verbose: bool = False) -> tuple[List[ScoredDoc], dict]:
        telemetry = {}

        start = time.perf_counter()
        vector_docs = retrieve_vector(self.db, self._normalize_scores, query, k=config.VECTOR_K)
        telemetry["vector_ms"] = round((time.perf_counter() - start) * 1000, 2)

        start = time.perf_counter()
        bm25_docs = retrieve_bm25(
            self.bm25,
            self.bm25_chunks,
            self._normalize_scores,
            query,
            k=config.BM25_K,
        )
        telemetry["bm25_ms"] = round((time.perf_counter() - start) * 1000, 2)

        start = time.perf_counter()
        merged = merge_results(
            vector_docs,
            bm25_docs,
            weights=self._weights(),
            feedback_priors=self.feedback_priors if config.ENABLE_FEEDBACK_PRIORS else {},
        )
        merged = merged[: config.MERGE_TOP_N]
        telemetry["merge_ms"] = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            print("\nHYBRID RETRIEVAL\n")
            for i, doc in enumerate(merged[:5], start=1):
                print(f"{i}. [{doc.source}] Hybrid Score: {doc.hybrid_score:.4f}")
                print(f"   {doc.content[:120]}...\n")

        return merged, telemetry

    def answer(self, query: str, verbose: bool = False, use_web: bool = False) -> dict:
        cache_key = f"{query}|{use_web}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        total_start = time.perf_counter()
        merged, telemetry = self.retrieve_candidates(query, verbose=verbose)

        start = time.perf_counter()
        top_merged = merged[: config.GRAPH_TOP_MERGED]
        expanded = expand_with_graph(
            self.graph,
            self.chunk_index,
            top_merged,
            max_neighbors=config.GRAPH_MAX_NEIGHBORS,
            max_chunks_per_neighbor=config.GRAPH_MAX_CHUNKS_PER_NEIGHBOR,
        )
        telemetry["graph_ms"] = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            print("\nGRAPH EXPANSION\n")
            print(f"Expanded from {len(top_merged)} -> {len(expanded)} chunks")

        start = time.perf_counter()
        reranked = rerank(self.reranker, query, expanded, top_k=config.RERANK_TOP_K)
        telemetry["rerank_ms"] = round((time.perf_counter() - start) * 1000, 2)

        if verbose:
            print("\nRERANKED RESULTS\n")
            for i, doc in enumerate(reranked, start=1):
                print(f"{i}. [{doc.source}] Rerank Score: {doc.rerank_score:.4f}")
                print(f"   {doc.content[:120]}...\n")

        context = self.build_context(reranked)

        prompt = f"""
You are an intelligent assistant reasoning over an Obsidian vault.

Rules:
1. Base your answer primarily on the provided context.
2. You may use general programming knowledge to interpret the user's query and the context.
3. Do NOT hallucinate information about the user's personal vault that is not in the context.
4. If context is insufficient, say: "I could not find this information in the Obsidian vault."

Context:
{context}

Question:
{query}

Answer:
"""

        start = time.perf_counter()
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            options={"temperature": 0},
            messages=[{"role": "user", "content": prompt}],
        )
        telemetry["generation_ms"] = round((time.perf_counter() - start) * 1000, 2)
        telemetry["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        answer_id = uuid.uuid4().hex
        result = {
            "answer_id": answer_id,
            "answer": response["message"]["content"],
            "sources": sorted({doc.source for doc in reranked}),
            "local_sources": sorted({doc.source for doc in reranked if doc.source_kind == "local"}),
            "web_sources": sorted({doc.source for doc in reranked if doc.source_kind == "web"}),
            "chunk_ids": [doc.doc_id for doc in reranked],
            "telemetry": telemetry,
        }

        self._cache_put(cache_key, result)
        return result

    def show_neighbors(self, note_query: str, depth: int = 1):
        node_map_lower = {node.lower(): node for node in self.graph.nodes()}
        results = self.node_db.similarity_search(note_query, k=3)

        if not results:
            print("\nNo matching notes found.")
            return

        print("\nSemantic Matches:\n")
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result.page_content}")

        matched_node = results[0].page_content
        actual_node = node_map_lower.get(matched_node.lower())

        if not actual_node:
            print(f"Error: '{matched_node}' not found in graph")
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

        if query.lower() in ["exit", "quit", "q", "/bye"]:
            break

        if query.lower() == "/verbose on":
            verbose = True
            print("Verbose mode enabled")
            continue

        if query.lower() == "/verbose off":
            verbose = False
            print("Verbose mode disabled")
            continue

        graph_match = re.match(r"/graph(\d+)?\s+(.+)", query)
        if graph_match:
            depth = int(graph_match.group(1)) if graph_match.group(1) else 1
            note = graph_match.group(2).strip()
            rag.show_neighbors(note, depth=depth)
            continue

        result = rag.answer(query, verbose=verbose)
        print("\nAnswer:\n")
        print(result["answer"])


if __name__ == "__main__":
    main()
