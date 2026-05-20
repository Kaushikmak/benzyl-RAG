import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
import pickle
import numpy as np
from sentence_transformers import CrossEncoder
import os
import re
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",)
logger = logging.getLogger(__name__)

@dataclass
class ScoredDoc:
    content: str
    doc: any
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""
    # RRF score matching
    @property
    def hybrid_score(self) -> float:
        return self.vector_score + self.bm25_score

class ImprovedRAG:
    def __init__(self):
        # EMBEDDING MODEL
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",model_kwargs={"device": "cuda"})

        # VECTOR DATABASE
        self.db = Chroma(persist_directory="./vectorstore",embedding_function=self.embedding_model)

        # NODE DATABASE
        self.node_db = Chroma(persist_directory="./node_vectorstore",embedding_function=self.embedding_model)
        
        # LOAD BM25
        with open("bm25.pkl", "rb") as f:
            self.bm25, self.bm25_chunks = pickle.load(f)
        
        # LOAD GRAPH
        with open("graph.pkl", "rb") as f:
            self.graph = pickle.load(f)
        
        # Build chunk index
        self.chunk_index = self._build_chunk_index()
        
        # RERANKER
        logger.info("Loading reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base",device="cuda")

        logger.info(f"Loaded {len(self.graph.nodes)} notes, {len(self.bm25_chunks)} chunks")
    
    def _build_chunk_index(self) -> Dict[str, List]:
        """Build index: note_name -> list of chunks from that note"""
        # give them default value
        index = defaultdict(list)
        for chunk in self.bm25_chunks:
            source = chunk.metadata.get("source", "")
            filename = os.path.basename(source)
            note_name = filename.replace(".md", "")
            index[note_name].append(chunk)
        return index
    
    def _normalize_scores(self, scores: List[float], k: int = 60) -> List[float]:
        """RRF: Reciprocal Rank Fusion normalization"""
        # Sort indices by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        
        # normalize scores using RRF 
        normalized = np.zeros(len(scores))
        for rank, idx in enumerate(ranked_indices, 1):
            normalized[idx] = 1.0 / (k + rank)
        
        return normalized.tolist()
    
    def retrieve_vector(self, query: str, k: int = 6) -> List[ScoredDoc]:
        """Vector similarity search with normalized scores"""
        results = self.db.similarity_search_with_relevance_scores(query, k=k)
        
        scores = [score for _, score in results]
        normalized = self._normalize_scores(scores)
        
        scored_docs = []
        for (doc, _), norm_score in zip(results, normalized):
            scored_docs.append(ScoredDoc(content=doc.page_content,doc=doc,vector_score=norm_score,source=os.path.basename(doc.metadata.get("source", "Unknown"))))
        
        return scored_docs
    
    def retrieve_bm25(self, query: str, k: int = 6) -> List[ScoredDoc]:
        """BM25 search with normalized scores"""
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        
        scores = [bm25_scores[idx] for idx in top_indices]
        normalized = self._normalize_scores(scores)
        
        scored_docs = []
        for idx, norm_score in zip(top_indices, normalized):
            doc = self.bm25_chunks[idx]
            scored_docs.append(ScoredDoc(
                content=doc.page_content,
                doc=doc,
                bm25_score=norm_score,
                source=os.path.basename(doc.metadata.get("source", "Unknown"))
            ))
        
        return scored_docs
    
    def merge_results(self, vector_docs: List[ScoredDoc], bm25_docs: List[ScoredDoc]) -> List[ScoredDoc]:
        """Merge vector and BM25 results with proper score fusion"""
        merged = {}
        
        # Merge by content
        for doc in vector_docs + bm25_docs:
            if doc.content not in merged:
                merged[doc.content] = doc
            else:
                # Combine scores
                merged[doc.content].vector_score += doc.vector_score
                merged[doc.content].bm25_score += doc.bm25_score
        
        # sort by hybrid score
        sorted_docs = sorted(merged.values(),key=lambda x: x.hybrid_score,reverse=True)
        
        return sorted_docs
    
    def expand_with_graph(self, docs: List[ScoredDoc], max_neighbors: int = 2,max_chunks_per_neighbor: int = 2) -> List[ScoredDoc]:
        """Expand results using graph neighbors"""
        expanded = []
        seen = set()
        
        for doc in docs:
            # add original doc
            if doc.content not in seen:
                expanded.append(doc)
                seen.add(doc.content)
            
            # find note name
            source = doc.doc.metadata.get("source", "")
            filename = os.path.basename(source)
            note_name = filename.replace(".md", "")
            
            # get neighbors from graph
            if note_name in self.graph:
                neighbors = list(self.graph.neighbors(note_name))[:max_neighbors]
                
                for neighbor in neighbors:
                    # chunk index for fast lookup
                    neighbor_chunks = self.chunk_index.get(neighbor, [])
                    
                    for chunk in neighbor_chunks[:max_chunks_per_neighbor]:
                        if chunk.page_content not in seen:
                            expanded.append(ScoredDoc(
                                content=chunk.page_content,
                                doc=chunk,
                                source=os.path.basename(chunk.metadata.get("source", "Unknown"))
                            ))
                            seen.add(chunk.page_content)
        
        return expanded
    
    def rerank(self, query: str, docs: List[ScoredDoc], top_k: int = 3) -> List[ScoredDoc]:
        """Rerank documents using cross-encoder"""
        if not docs:
            return []
        
        pairs = [(query, doc.content) for doc in docs]
        scores = self.reranker.predict(pairs)
        
        for doc, score in zip(docs, scores):
            doc.rerank_score = float(score)
        
        reranked = sorted(docs, key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_k]
    
    def build_context(self, docs: List[ScoredDoc]) -> str:
        """Build context string from documents"""
        parts = []
        for doc in docs:
            parts.append(f"""SOURCE: {doc.source}

                CONTENT:
                {doc.content}"""
            )
            
            return "\n\n" + "="*50 + "\n\n".join(parts)
        
        def answer(self, query: str, verbose: bool = False) -> str:
            """Main RAG pipeline"""
            # Hybrid retrieval
            vector_docs = self.retrieve_vector(query, k=6)
            bm25_docs = self.retrieve_bm25(query, k=6)
            merged = self.merge_results(vector_docs, bm25_docs)
            
            if verbose:
                print("\n=== HYBRID RETRIEVAL ===")
                for i, doc in enumerate(merged[:5], 1):
                    print(f"{i}. [{doc.source}] Score: {doc.hybrid_score:.3f}")
                    print(f"   {doc.content[:100]}...\n")
            
            # Graph expansion
            top_merged = merged[:4]
            expanded = self.expand_with_graph(top_merged, max_neighbors=2, max_chunks_per_neighbor=2)
            
            if verbose:
                print(f"\n=== GRAPH EXPANSION ===")
                print(f"Expanded from {len(top_merged)} to {len(expanded)} chunks")
            
            # Reranking
            reranked = self.rerank(query, expanded, top_k=3)
            
            if verbose:
                print("\n=== RERANKED RESULTS ===")
                for i, doc in enumerate(reranked, 1):
                    print(f"{i}. [{doc.source}] Rerank: {doc.rerank_score:.3f}")
                    print(f"   {doc.content[:100]}...\n")
            
            # context
            context = self.build_context(reranked)
            
            prompt = f"""You are a strict retrieval-based assistant.

                    Rules:
                    1. Answer ONLY using the provided context.
                    2. Do NOT use external knowledge.
                    3. Do NOT hallucinate.
                    4. If answer is not explicitly present, reply EXACTLY:
                    "I could not find this information in the Obsidian vault."

                    Context:
                    {context}

                    Question: {query}

                    Answer:"""
        
            response = ollama.chat(
                model="qwen3:8b",
                options={"temperature": 0},
                messages=[{"role": "user", "content": prompt}]
            )
        
            return response["message"]["content"]
    
    def show_neighbors(self, note_query: str, depth: int = 1):
        """Show graph neighbors of a note"""
        node_map_lower = {node.lower(): node for node in self.graph.nodes()}
        
        results = self.node_db.similarity_search(note_query, k=3)
        if not results:
            print("\nNo matching notes found.")
            return
        
        print("\nSemantic Matches:\n")
        for idx, result in enumerate(results):
            print(f"{idx+1}. {result.page_content}")

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
            print(f"\nDepth {d+1}:\n")

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
    print("  /verbose on/off  - Toggle detailed output")
    print("  /graph[N] <note> - Show graph neighbors (depth N)")
    print("  exit quit q /bye - Exit")
    
    verbose = False
    
    while True:
        query = input("\nAsk: ").strip()
        
        # exit commands
        if query.lower() in ["exit", "quit", "q", "/bye"]:
            break
        
        # verborse toggle
        if query.lower() == "/verbose on":
            verbose = True
            print("Verbose mode enabled")
            continue
        elif query.lower() == "/verbose off":
            verbose = False
            print("Verbose mode disabled")
            continue
        
        # graph command
        graph_match = re.match(r"/graph(\d+)?\s+(.+)", query)
        if graph_match:
            depth = int(graph_match.group(1)) if graph_match.group(1) else 1
            note = graph_match.group(2).strip()
            rag.show_neighbors(note, depth=depth)
            continue
        
        # output
        answer = rag.answer(query, verbose=verbose)
        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()