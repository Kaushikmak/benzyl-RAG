# Obsidian Hybrid Graph RAG

A powerful, local-first Retrieval-Augmented Generation (RAG) assistant specifically designed to chat with and explore your Obsidian Vault. 

This application combines multiple advanced retrieval strategies to give you highly accurate answers based purely on your personal notes:
- **Vector Search**: Semantic similarity using ChromaDB and `BAAI/bge-m3` embeddings.
- **Keyword Search (BM25)**: Fast, exact-match text search.
- **Graph Traversal**: Explores the logical links and connections between your Obsidian notes.
- **Reranking**: Refines and scores the retrieved context using `BAAI/bge-reranker-v2-m3`.
- **Local LLMs**: Entirely private generation using **Ollama**.

It features a robust CLI interface and core Python RAG library with conversational history, source-tracking, and feedback controls.

---

## Project Structure
```text
obsidianRAG/
├── app/                  # Core RAG logic, config, and retrieval modules
├── data/                 # Stores your Obsidian Vault, graphs, and BM25 chunks
├── indexing/             # Pipeline for chunking notes, building vectors & graphs
├── scripts/              # Helper scripts
├── vectorstore/          # ChromaDB vector database files
└── main.py               # Single unified CLI entrypoint
```

---

##  Setup & Usage (Local Python Environment)

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed on your host machine.

### 2. Configuration
1. Place your Obsidian vault folder inside `data/obsidianVault/` (or update `VAULT_PATH` in `app/config.py`).
2. Make sure Ollama is running and pull your preferred model (e.g., `ollama pull qwen3:8b` or whatever is defined as `OLLAMA_MODEL` in `app/config.py`).

### 3. Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Indexing Your Vault
Before you can chat, you must index your Obsidian vault. This processes your Markdown files into vector embeddings, keyword chunks, and a knowledge graph.
```bash
python main.py index
```
*Note: You need to re-run this command whenever you significantly update your Obsidian notes.*

### 5. Running the Application
Chat with your vault directly in the terminal:
```bash
python main.py cli
```

---

## Configuration Details
All major settings are located in `app/config.py`:
- `OLLAMA_MODEL`: The local LLM to use (default setup implies a Qwen model, but Llama 3 or Mistral work perfectly).
- `EMBED_MODEL`: The HuggingFace embedding model (default: `BAAI/bge-m3`).
- `RERANK_MODEL`: The cross-encoder used for reranking (default: `BAAI/bge-reranker-v2-m3`).
- Chunking limits, vector store paths, and device settings (`cuda` vs `cpu`) can also be toggled here.
