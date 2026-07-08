# Obsidian Hybrid Graph RAG Architecture

This document provides a detailed breakdown of the internal architecture and data flow of the Obsidian RAG model. It outlines how a user's query is processed from the moment it is asked in the CLI or WebApp, all the way to the final generated response, relying strictly on the local Obsidian vault.

---

## 1. System Overview

The system is a local-first Retrieval-Augmented Generation (RAG) pipeline designed specifically for Obsidian vaults. It utilizes a **multi-stage retrieval architecture** that fuses semantic search, keyword search, and knowledge graph traversal, followed by cross-encoder reranking and local generation via Ollama.

The application operates primarily via **CLI Mode** (`python main.py cli`) or programmatically importing the `ImprovedRAG` class.

---

## 2. The Indexing Phase (Pre-computation)

Before any query can be answered, the vault must be indexed (`python main.py index`). The indexing pipeline creates four distinct data structures:

1. **Chunk Index**: Large markdown files are split into smaller, overlapping chunks to preserve context without exceeding LLM token limits.
2. **Vector Store (ChromaDB)**: Every chunk is converted into a dense vector embedding using `BAAI/bge-m3`. This allows for semantic similarity matching.
3. **BM25 Index**: A classic inverted index is built using the BM25 algorithm. This is crucial for exact keyword matches (e.g., specific names, acronyms, or code snippets).
4. **Knowledge Graph**: The indexer parses Obsidian `[[wikilinks]]` to build a directed graph (using NetworkX). Nodes represent files, and edges represent links between files.

---

## 3. The Retrieval Pipeline

When a user submits a query, the backend executes the following multi-stage pipeline:

### Stage 3.1: Parallel Baseline Retrieval
The query is simultaneously passed to both the Vector database and the BM25 index.

- **Vector Retrieval**: Computes the cosine similarity between the query embedding and chunk embeddings, returning the top $k$ candidates.
- **BM25 Retrieval**: Computes term-frequency/inverse-document-frequency (TF-IDF) based keyword matching, returning the top $k$ candidates.

### Stage 3.2: Score Normalization (Reciprocal Rank Fusion)
Because Vector cosine scores (typically 0 to 1) and BM25 scores (unbounded) are not directly comparable, the system normalizes them using a variation of **Reciprocal Rank Fusion (RRF)**. 

For a given list of scores ranked from highest to lowest, the normalized score for the item at rank $r$ is:

$$ \text{Normalized Score} = \frac{1}{k_{rrf} + r} $$

*(Where $k_{rrf}$ is a smoothing constant, default `60`, and $r \ge 1$)*.

### Stage 3.3: Hybrid Merge & Feedback Fusion
The normalized results from both retrievers are merged. If a chunk is found in both, its scores are summed. The final **Hybrid Score** is calculated using weighted coefficients and historical user feedback priors:

$$ S_{hybrid} = (S_{vector} \times W_{vector}) + (S_{bm25} \times W_{bm25}) + (P_{feedback} \times W_{feedback}) $$

- $W_{vector}, W_{bm25}$: Configurable weight tunings.
- $P_{feedback}$: A historical prior based on whether previous users voted "Correct" or "Not Correct" on answers that utilized this specific chunk.

The candidates are sorted by $S_{hybrid}$, and the top $N$ chunks are promoted to the next stage.

### Stage 3.4: Graph Expansion
To capture broader context (e.g., "What does this note relate to?"), the top $N$ hybrid chunks are mapped to their parent Obsidian files in the **Knowledge Graph**.

The algorithm traverses the graph up to `max_neighbors` steps. For each neighboring file discovered via a `[[wikilink]]`, the system blindly pulls a subset of its chunks (`max_chunks_per_neighbor`) into the candidate pool. 

*Formulaic representation of the pool size:*
$$ C_{pool} = C_{hybrid\_top} \cup \left( \bigcup_{i=1}^{neighbors} C_{neighbor\_chunks\_subset} \right) $$

### Stage 3.5: Cross-Encoder Reranking
Graph expansion introduces noise (irrelevant chunks from loosely related files). To fix this, the entire candidate pool is passed through a **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`).

Unlike standard embeddings which process the query and document separately, a cross-encoder processes the tuple `(Query, Document)` together, utilizing full self-attention across both texts. It outputs a highly accurate logits score predicting relevance.

The pool is sorted by $S_{rerank}$, and only the absolute top chunks (e.g., Top 5) are passed to the generator.

---

## 4. The Generation Phase

The strictly filtered, highly relevant chunks are formatted into a system prompt. The prompt explicitly instructs the LLM to base its answer *only* on the provided context.

### The Ollama Integration
The prompt is sent to the local LLM via the Ollama API. In this architecture, the LLM is instructed to utilize a "Chain of Thought" structure:

```xml
<thinking>
1. Identify key constraints from the prompt.
2. Analyze Chunk A and Chunk B.
3. Formulate the response.
</thinking>
<answer>
The synthesized answer goes here...
</answer>
```

### Streaming Responses
The `ImprovedRAG.answer_stream()` method yields token events generator-style, allowing interactive CLI or programmatic consumers to stream tokens as they are generated by Ollama.

---

## 5. Feedback Loop
When generation is complete, feedback can be stored into the `conversations.db` SQLite database using `app.feedback.save_feedback()`.

A helper function recalculates the Bayesian feedback priors ($P_{feedback}$) for specific chunk IDs cited in that answer. The next time a query is evaluated, the `Hybrid Merge` (Stage 3.3) mathematically favors or penalizes chunks based on historical interactions.
