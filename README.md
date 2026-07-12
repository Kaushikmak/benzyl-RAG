# benzyl-RAG

A powerful, local-first Retrieval-Augmented Generation (RAG) assistant specifically designed to chat with and explore your document repositories.

This application is architected around a **6-Stage Production RAG Track** to deliver resilient, highly accurate answers from your documents:

1. **Ingestion & Parsing**: Universal translator handling messy documents via **Apache Tika**, **Unstructured**, **Docling**, and multi-tier **PDF OCR Fallback** (`pdf2image` + `pytesseract` for image-only/scanned PDFs lacking a text layer).
2. **Intelligent Chunking & Metadata**: Sentence window / hierarchical splitting that preserves heading breadcrumbs, treats tables as atomic units, and precomputes summaries, keywords, and hypothetical questions.
3. **Database Storage & Hybrid Retrieval**: Multi-step funnel combining **Metadata/Payload Filtering (Qdrant native filters)**, **Qdrant** dense vector search (`BAAI/bge-m3`), **BM25** exact keyword matching, and **Cross-Encoder Reranking** (`BAAI/bge-reranker-v2-m3`).
4. **Orchestration & Routing**: Fast-pass **Query Router** (`app/router.py`) classifying requests to bypass heavy DB lookups for deterministic math, status, or conversational queries.
5. **Multi-Agent Execution & Safety (`app/agents/`)**: 16-agent Layer 5 Enterprise Orchestration Engine running a 14-stage execution sequence (`Security` -> `Planner` -> `Rewrite` -> `Cache` -> `Researcher` -> `Security Audit` -> `Compression` -> `Reranker` -> `Citation` -> `Synthesis` -> `Reflection` -> `Verification` -> `Formatter` -> `Observability`), atomic filesystem operations (`.tmp` -> flush -> fsync -> os.replace), and persistent Human-in-the-Loop (`HITL`) state machine (`.mission_state/`).
6. **Evaluation & Monitoring**: Real-time **Continuous Evaluator (Heuristic Report Card)** grading heuristic faithfulness (N-gram Jaccard entailment), relevance (sigmoid cross-encoder logit), answer overlap, latency, and operational cost.

It features a streamlined Python CLI interface (`python main.py cli` or `python main.py query -q "..."`) with conversational history, source tracking, and continuous report card telemetry.

---

## Project Structure

```text
benzyl-RAG/
├── app/                  # Core RAG logic, config, and retrieval modules
│   └── agents/           # Layer 5: 16-agent orchestration engine, Pydantic v2 models & HITL workflows
├── data/                 # Universal files (.pdf, .docx, .xlsx, .html, .txt, .md, etc.)
├── .data/                # Hidden internal storage (Qdrant DB, BM25/Graph indexes, provenance & evaluation logs)
├── .mission_state/       # Persistent serialized MissionSnapshot files for HITL approvals
├── indexing/             # Pipeline for chunking documents, building vectors & relationship graphs
├── guardrails/           # Production security and adversarial defense pipeline
├── scripts/              # Helper scripts
└── main.py               # Single unified CLI entrypoint
```

---

## Agent Reference (Layer 5 Engine)

Every query runs through a deterministic 14-stage pipeline across 16 agents. Most run automatically in the background (security scanning, retrieval, reranking, citation, reflection, verification, formatting). A few respond to direct user intent:

| Agent          | Role                                                                                        | Example                                                     |
| -------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Planner**    | Decomposes multi-part queries into a prioritized retrieval plan                             | `"Compare GFS and HDFS chunk replication"`                  |
| **Researcher** | Runs the 4-stage retrieval funnel (metadata filter -> hybrid RRF -> graph expansion -> rerank) | `"What are the assumptions of the GFS paper?"`              |
| **Synthesis**  | Calls local `qwen2.5:7b` to generate the answer from retrieved context                      | `"Explain all layers of blockchain with examples"`          |
| **File**       | Handles SAVE/APPEND/DELETE with atomic writes and HITL approval                             | `"Summarise the GFS conclusion and save to gfs_summary.md"` |
| **Math**       | Sandboxed AST evaluator for arithmetic — no LLM call                                        | `"145 * 23 + sqrt(144)"`                                    |
| **Status**     | Answers system introspection from cached telemetry                                          | `"How many documents are indexed?"`                         |

> **HITL Safety Gate**: Any SAVE or DELETE creates a `HITLApprovalRequest` (`PENDING`) in `.mission_state/<id>.json`. Nothing is written until explicitly approved.

---

## Setup & Usage (Local Python Environment)

### 1. Prerequisites

- Python 3.10+
- System packages: `tesseract` (`/usr/bin/tesseract`) and `poppler-utils` (required for image-only PDF page rasterization and OCR fallback via `pdf2image` and `pytesseract`)
- [Ollama](https://ollama.com/) installed on your host machine.

### 2. Configuration

1. Place your documents (`.pdf`, `.docx`, `.xlsx`, `.html`, `.txt`, `.md`, etc.) inside `data/`.
2. Make sure Ollama is running and pull your preferred model (e.g., `ollama pull qwen2.5:7b` or whatever is defined as `OLLAMA_MODEL` in `app/config.py`).

### 3. Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Indexing Your Documents

Before you can chat, you must index your documents. This processes your files into vector embeddings, keyword chunks, and a document relationship graph. Scanned or image-only PDFs (such as ID cards without a text layer) are automatically rasterized via `pdf2image` and OCR'd with `pytesseract`.

```bash
python main.py index
```

_Note: Re-running `python main.py index` rebuilds the Qdrant collections (`document_chunks` & `document_nodes`) cleanly (`force_recreate=True`), preventing stale or duplicate entries across reindexing runs._

### 5. Running the Application & Inspecting Guardrails Telemetry

Use the unified pipeline runner script `scripts/run_pipeline.sh`:

```bash
# 1. Interactive CLI mode
./scripts/run_pipeline.sh cli

# 2. One-shot query through Guardrails defenses
./scripts/run_pipeline.sh query -q "How does the authentication middleware work?"

# 3. Evaluate Guardrails defense benchmarks (ASR & Rank Shift table)
./scripts/run_pipeline.sh eval

# 4. Run automated unit tests
./scripts/run_pipeline.sh test
```

---

## Configuration Details

All major settings are located in `app/config.py`:

- `OLLAMA_MODEL`: The canonical local LLM to use (default: `qwen2.5:7b`; alternates such as `qwen3:8b` or `llama3` work seamlessly).
- `EMBED_MODEL`: The HuggingFace embedding model (default: `BAAI/bge-m3`).
- `RERANK_MODEL`: The cross-encoder used for reranking (default: `BAAI/bge-reranker-v2-m3`).
- Chunking limits, vector store paths, and device settings (`cuda` vs `cpu`) can also be toggled here.

---

## Architecture Overview

### Guardrails Pipeline (`guardrails/`)

Guardrails hardens the benzyl-RAG pipeline against two principal threat vectors (aligned with OWASP LLM Top 10 framing):

1. **Corpus Poisoning**: Adversaries inject near-duplicate or semantic outlier notes to manipulate retrieval rankings and dominate the context window.
2. **Indirect Prompt Injection**: Malicious instructions embedded in note contents (`Ignore all previous instructions...`) attempt to hijack the LLM generation stage.

```text
guardrails/
├── core/                  # Retriever-agnostic interfaces (Candidate, BaseRetriever, DefenseStage)
├── redteaming/            # Corpus poisoning & indirect prompt injection payload generators
├── defenses/
│   ├── detection/         # Near-duplicate clusterer, embedding outlier detector, graph-expansion gate
│   ├── filtering/         # Prompt sanitization, Unicode/HTML normalization, data block quarantine
│   └── isolation/         # RobustRAG-style partition-generate-aggregate evaluation mode
├── evaluation/            # Evaluation harness measuring ASR, ΔnDCG, latency, and safety net clone
└── pipeline.py            # GuardrailsPipeline wrapping hybrid retrieval
```

### Evaluation Progression & Benchmark Results

All attack and evaluation experiments run inside a disposable clone (`.data/clone/`) to preserve live repository integrity.

> **Methodology**: Rows are **cumulative** — each row stacks all defenses from previous rows plus one new layer, matching how the production RAGShield pipeline runs. **N = 60 attack payloads per row** (30 corpus-poisoning + 30 indirect-injection payloads spanning 10 distinct techniques) + 30 clean candidates = **90 total evaluated candidates**.

| Pipeline Configuration (Cumulative) | ASR (successes/N) |  ASR (%)  |    ΔnDCG    | Latency Overhead | Key Defense Mechanism                                                                    |
| :---------------------------------- | :---------------: | :-------: | :---------: | :--------------: | :--------------------------------------------------------------------------------------- |
| **0. Baseline (No Guardrails)**     |     **60/60**     | **85.7%** |   -1.0000   |      0.0 ms      | Ungated link traversal & unnormalized injection adoption                                 |
| **1. + Detection Layer**            |     **20/60**     | **66.7%** |   -1.0000   |     +2.8 ms      | Near-duplicate deduplication (Jaccard $\ge 0.82$) & outlier quarantine ($z > 2.0\sigma$) |
| **2. + Graph-Expansion Gate**       |     **13/60**     | **56.5%** |   -1.0000   |     +1.2 ms      | Rejects graph neighbors below query similarity floor ($T_{graph} \ge 0.25$)              |
| **3. + Prompt Sanitization**        |     **13/60**     | **56.5%** |   -1.0000   |     +3.5 ms      | Strips unicode confusables, quarantines injection regexes, frames untrusted data         |
| **4. + Isolation Mode**             |     **0/60**      | **0.0%**  | **+0.0000** |    +420.0 ms     | RobustRAG partition-and-aggregate multi-generation consensus                             |

### Limitations

Filter, detection, and isolation defenses assume adversarial content can be distinguished by similarity thresholding, clustering anomalies, or structural injection signatures. Under adaptive or pipeline-robust adversaries who mimic legitimate semantic distributions and avoid known syntax patterns, heuristic detection may degrade. Continuous evaluation and guardrail filtering provide layered defense-in-depth.
