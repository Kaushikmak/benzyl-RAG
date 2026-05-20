from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
import pickle
import numpy as np
from sentence_transformers import CrossEncoder
import networkx as nx
import os
import re

 
# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# VECTOR DATABASE
db = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embedding_model
)

# NODE DATABASE
node_db = Chroma(
    persist_directory="./node_vectorstore",
    embedding_function=embedding_model
)
 
# LOAD BM25
with open("bm25.pkl", "rb") as f:
    bm25, bm25_chunks = pickle.load(f)
 
# LOAD GRAPH
with open("graph.pkl", "rb") as f:
    graph = pickle.load(f)

 
# RERANKER
print("Loading reranker...")
reranker = CrossEncoder(
    "BAAI/bge-reranker-base",
    device="cuda"
)
 
# GRAPH FUNCTIONS
def show_neighbors(note_query, depth=1):
    # Create lowercase mapping for case-insensitive node lookup
    node_map_lower = {node.lower(): node for node in graph.nodes()}
    
    results = node_db.similarity_search(note_query, k=3)
    if not results:
        print("\nNo matching notes found.")
        return
    
    print("\nSemantic Matches:\n")
    for idx, result in enumerate(results):
        print(f"{idx+1}. {result.page_content}")

    matched_node = results[0].page_content
    
    # Case-insensitive lookup
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
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    print(f"{node} -> {neighbor}")
                    next_nodes.add(neighbor)
                    visited.add(neighbor)

        current = next_nodes
 


# MAIN LOOP
while True:
    query = input("\nAsk: ")

    # EXIT COMMANDS
    if query.lower() in ["exit","quit","q","/bye"]:
        break

    # GRAPH COMMAND
    graph_match = re.match(r"/graph(\d+)?\s+(.+)", query)
    if graph_match:
        depth = int(graph_match.group(1)) if graph_match.group(1) else 1
        note = graph_match.group(2).strip()
        show_neighbors(note, depth=depth)
        continue
     
    # VECTOR SEARCH
    vector_results = db.similarity_search_with_relevance_scores(query,k=6)
    vector_docs = []

    for doc, score in vector_results:
        vector_docs.append((doc, score))

     
    # BM25 SEARCH
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:6]
    bm25_docs = []

    for idx in top_bm25_indices:
        bm25_docs.append((bm25_chunks[idx],bm25_scores[idx]))

     
    # HYBRID MERGING
    combined_docs = {}

    # VECTOR RESULTS
    for doc, score in vector_docs:
        content = doc.page_content
        if content not in combined_docs:
            combined_docs[content] = {"doc": doc,"score": 0}
        combined_docs[content]["score"] += float(score)

    # BM25 RESULTS
    for doc, score in bm25_docs:
        content = doc.page_content
        if content not in combined_docs:
            combined_docs[content] = {"doc": doc,"score": 0}
        combined_docs[content]["score"] += float(score)

     
    # SORT HYBRID RESULTS
    final_docs = sorted(combined_docs.values(),key=lambda x: x["score"],reverse=True)
    final_docs = final_docs[:4]
     
    # GRAPH EXPANSION
    expanded_docs = []
    seen_chunks = set()

    for item in final_docs:
        doc = item["doc"]
        # ADD MAIN DOC
        if doc.page_content not in seen_chunks:
            expanded_docs.append(doc)
            seen_chunks.add(doc.page_content)

        # FIND GRAPH NEIGHBORS
        source = doc.metadata.get("source","")
        filename = os.path.basename(source)
        note_name = filename.replace(".md","")

        if note_name in graph:
            neighbors = list(graph.neighbors(note_name))
            for neighbor in neighbors[:3]:
                for chunk in bm25_chunks:
                    chunk_source = chunk.metadata.get("source","")
                    chunk_filename = os.path.basename(chunk_source)
                    chunk_note = chunk_filename.replace(".md","")
                    if chunk_note == neighbor:
                        if chunk.page_content not in seen_chunks:
                            expanded_docs.append(chunk)
                            seen_chunks.add(chunk.page_content)

     
    # RERANKING
    rerank_pairs = [(query,doc.page_content)for doc in expanded_docs]

    rerank_scores = reranker.predict(rerank_pairs)
    reranked = []

    for doc, score in zip(expanded_docs,rerank_scores):
        reranked.append({"doc": doc,"score": float(score)})

    reranked = sorted(reranked,key=lambda x: x["score"],reverse=True)
    reranked = reranked[:3]
     
    # CONTEXT BUILDING
    context_parts = []

    for item in reranked:

        doc = item["doc"]
        source = doc.metadata.get("source","Unknown")
        filename = os.path.basename(source)

        context_parts.append(
                        f"""
                        SOURCE: {filename}

                        CONTENT:
                        {doc.page_content}
                        """
        )
    context = "\n\n".join(context_parts)

     
    # PROMPT
    prompt = f"""
        You are a strict retrieval-based assistant.

        Rules:
        1. Answer ONLY using the provided context.
        2. Do NOT use external knowledge.
        3. Do NOT hallucinate.
        4. If answer is not explicitly present,
        reply EXACTLY:
        "I could not find this information in the Obsidian vault."

        Context:
        ----------------
        {context}
        ----------------

        Question:
        {query}

        Answer:
        """

    # GENERATION
    response = ollama.chat(
        model="qwen3:8b",
        options={"temperature": 0},
        messages=[{"role": "user","content": prompt}]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])