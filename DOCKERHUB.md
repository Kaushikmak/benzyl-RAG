# Benzyl RAG

**Agentic AI Assistant for Private Document Collections.**

Benzyl RAG is a RAG (Retrieval-Augmented Generation) pipeline that parses multi-column layouts, tables, and scanned PDFs, indexes them into a hybrid vector + lexical index, and synthesizes accurate answers using local LLMs—without sending sensitive documents to the cloud.

## Key Features

- **Structured Layout Parsing (`Docling` + `Apache Tika`)**: Preserves visual hierarchy, multi-column layouts, section headings, and complex markdown tables across `.pdf`, `.docx`, `.xlsx`, `.html`, `.txt`, and more.
- **OCR Fallback (`tesseract` + `poppler-utils`)**: Automatically rasterizes and OCRs scanned or image-only PDFs lacking a text layer.
- **Hybrid Retrieval (`Qdrant` + `BM25` + `RRF`)**: Combines 1024-dimensional dense semantic embeddings (`BAAI/bge-m3`) with sparse exact keyword matching (`Rank-BM25`), normalized via Reciprocal Rank Fusion ($k_{rrf}=60$) and rescored with a Cross-Encoder reranker (`BAAI/bge-reranker-v2-m3`).
- **Adversarial Guardrails**: Defends against OWASP LLM Top 10 threats including **Corpus Poisoning** (near-duplicate clustering & outlier quarantine) and **Indirect Prompt Injection** (Unicode normalization & untrusted data block framing).
- **Human-In-The-Loop (HITL) Safety Gate**: Gated filesystem operations (`SAVE`, `APPEND`, `DELETE`) block mutations until explicitly approved.
- **Local-First Privacy**: Runs against a local [Ollama](https://ollama.com/) daemon (`qwen2.5:7b`). Your documents and queries never leave your machine.

## Quickstart with Docker Compose

Create a `docker-compose.yaml` file on your host machine:

```yaml
name: benzyl-rag

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: benzyl-rag-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage

  rag-app:
    image: tastytaco/benzyl-rag:latest
    container_name: benzyl-rag-app
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - DATA_DIR=/app/data
      - OLLAMA_URL=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./data:/app/data
      - ./.data:/app/.data
      - ./.mission_state:/app/.mission_state
    stdin_open: true
    tty: true
```

### 1. Index Your Documents

Place your files (`.pdf`, `.docx`, etc.) inside `./data/`, ensure Ollama is running on your host machine, and run:

```bash
docker compose run --rm rag-app python main.py index
```

### 2. Query Your Documents

Ask questions against your indexed knowledge base:

```bash
docker compose run --rm rag-app python main.py cli
```

Or run one-shot CLI queries:

```bash
docker compose run --rm rag-app python main.py query -q "What is the authentication flow?"
```

---

## Volume Mounts & Persistence

| Host Path          | Container Path        | Purpose                                                             |
| :----------------- | :-------------------- | :------------------------------------------------------------------ |
| `./data`           | `/app/data`           | Source directory containing your raw documents to be indexed        |
| `./.data`          | `/app/.data`          | Cached BM25 indices, metadata summaries, and graph relationships    |
| `./.mission_state` | `/app/.mission_state` | Persistent state for Human-in-the-Loop (HITL) file action approvals |

---

## License & Source Code

- **GitHub Repository**: [Kaushikmak/benzyl-RAG](https://github.com/Kaushikmak/benzyl-RAG)
