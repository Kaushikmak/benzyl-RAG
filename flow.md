# End-to-End Architecture & Execution Flow

> *A complete architectural walkthrough covering document ingestion, intelligent processing, hybrid retrieval, query routing, agent orchestration, human-in-the-loop approvals, and continuous evaluation.*



## System Architecture Flowchart

This flowchart maps all six core architectural stages of **benzyl-RAG** and illustrates data flow from raw external documents to synthesized grounded responses and evaluation report cards.

```mermaid
flowchart TD
    %% ==========================================
    %% STAGE 1: INGESTION & PARSING
    %% ==========================================
    subgraph STAGE_1 ["Stage 1: Universal Ingestion & Parsing"]
        RawDocs["External Documents (.pdf, .docx, .xlsx, .html, .md, .txt, images)"]
        RouterIngest{"Document Format Router"}

        DoclingParser["Docling Parser"]
        OCRFallback["Multi-Tier OCR Fallback"]
        TikaParser["Apache Tika Universal Parser"]
        UnstructuredStruct["Unstructured Partitioner"]
        GarbageGate{"Garbage-Text Safety Gate"}
    end

    %% ==========================================
    %% STAGE 2: INTELLIGENT CHUNKING & METADATA
    %% ==========================================
    subgraph STAGE_2 ["Stage 2: Intelligent Chunking & Metadata Extraction"]
        HeadingStack["Heading Stack Breadcrumb Tracker"]
        TablePreserver["Atomic Table Preserver"]
        SentenceSplitter["Sentence Window Splitting"]
        MetadataPrecompute["Metadata Precomputation"]
    end

    %% ==========================================
    %% STAGE 3: DATABASE STORAGE & HYBRID RETRIEVAL
    %% ==========================================
    subgraph STAGE_3 ["Stage 3: Database Storage & Hybrid Retrieval Funnel"]
        QdrantStore[("Qdrant Vector DB (BAAI/bge-m3 HNSW Index d=1024)")]
        BM25Store[("Rank-BM25 Inverted Keyword Index (.data/bm25.pkl)")]
        GraphStore[("NetworkX Adjacency Graph (.data/graph.pkl)")]

        FunnelS1["Stage 1: Hard Metadata & Payload Filtering (Qdrant Native Filters)"]
        FunnelS2["Stage 2: Wide Hybrid RRF Fusion"]
        FunnelS3["Stage 3: Bounded Graph Neighborhood Expansion"]
        FunnelS4["Stage 4: Precision Cross-Encoder Reranking"]
    end

    %% ==========================================
    %% STAGE 4: ORCHESTRATION & ROUTING
    %% ==========================================
    subgraph STAGE_4 ["Stage 4: Fast-Pass Orchestration & Semantic Query Routing"]
        UserQuery["Query Input"]
        QueryRouter{"Semantic Query Router"}
        MathFast["DIRECT_MATH: Sandboxed AST Arithmetic Evaluator"]
        StatusFast["DIRECT_STATUS: In-Memory Telemetry"]
        ConvFast["DIRECT_CONVERSATIONAL: Direct Salutation Response"]
    end

    %% ==========================================
    %% STAGE 5: MULTI-AGENT EXECUTION & SAFETY
    %% ==========================================
    subgraph STAGE_5 ["Stage 5: Multi-Agent Execution Engine (app/agents/)"]
        AgentOrch["AgentOrchestrator specialized single-responsibility agents"]

        subgraph Pipeline14 ["multi-Stage Execution Sequence"]
            direction LR
            S_Sec["1. Security"] --> S_Plan["2. Planner"] --> S_Rew["3. Rewrite"] --> S_Cache["4. Cache"] --> S_Res["5. Researcher"]
            S_Res --> S_Audit["6. Security Audit"] --> S_Comp["7. Compression"] --> S_Rerank["8. Reranker"] --> S_Cit["9. Citation"]
            S_Cit --> S_Synth["10. Synthesis"] --> S_Refl["11. Reflection"] --> S_Verif["12. Verification"] --> S_Form["13. Formatter"] --> S_Obs["14. Observability"]
        end

        HITLGate{"HITL Mutation Gate (SAVE / DELETE Actions)"}
        MissionSnapshot[("Persistent HITL MissionSnapshot State (.mission_state/{id}.json")]
        FileAgentAtomic["FileAgent Atomic Filesystem Operations"]
        LocalLLM["Local Ollama Synthesis Runtime (qwen2.5:7b / qwen3:8b)"]
    end

    %% ==========================================
    %% STAGE 6: EVALUATION & MONITORING
    %% ==========================================
    subgraph STAGE_6 ["Stage 6: Continuous Evaluation & Monitoring"]
        ReportCard["Heuristic Report Card"]
        MetricFaith["Faithfulness: N-gram Jaccard Entailment (Flagged if < 0.40)"]
        MetricRel["Relevance: Sigmoid Cross-Encoder Logit Transformation"]
        MetricCost["Production Cloud Cost Estimator (GPT-4o vs Claude 3.5 vs Local Compute)"]
        EvalHistory[("Asynchronous Evaluation History Log")]
    end

    %% Flow Connections: Stage 1 -> Stage 2 -> Stage 3 (Indexing)
    RawDocs --> RouterIngest
    RouterIngest -- "PDF" --> DoclingParser
    RouterIngest -- "Image-Only / Scanned PDF" --> OCRFallback
    RouterIngest -- "Non-PDF / Office / Web" --> TikaParser
    DoclingParser --> UnstructuredStruct
    OCRFallback --> UnstructuredStruct
    TikaParser --> UnstructuredStruct
    UnstructuredStruct --> GarbageGate
    GarbageGate -- "Valid Text" --> HeadingStack
    HeadingStack --> TablePreserver
    TablePreserver --> SentenceSplitter
    SentenceSplitter --> MetadataPrecompute

    MetadataPrecompute --> QdrantStore
    MetadataPrecompute --> BM25Store
    MetadataPrecompute --> GraphStore

    %% Flow Connections: Stage 4 -> Stage 3 / Fast-Pass
    UserQuery --> QueryRouter
    QueryRouter -- "DIRECT_MATH" --> MathFast
    QueryRouter -- "DIRECT_STATUS" --> StatusFast
    QueryRouter -- "DIRECT_CONVERSATIONAL" --> ConvFast
    QueryRouter -- "VAULT_RAG (Domain Query)" --> FunnelS1

    FunnelS1 --> FunnelS2
    FunnelS2 --> FunnelS3
    FunnelS3 --> FunnelS4

    %% Flow Connections: Stage 3 -> Stage 5 (Orchestration)
    FunnelS4 --> AgentOrch
    AgentOrch --> Pipeline14
    S_Synth --> LocalLLM
    AgentOrch -- "Mutation Request (FileAgent)" --> HITLGate
    HITLGate -- "status = PENDING" --> MissionSnapshot
    HITLGate -- "APPROVED" --> FileAgentAtomic

    %% Flow Connections: Stage 5 -> Stage 6 (Evaluation)
    LocalLLM --> ReportCard
    FileAgentAtomic --> ReportCard
    ReportCard --> MetricFaith
    ReportCard --> MetricRel
    ReportCard --> MetricCost
    ReportCard --> EvalHistory
```


## Sequence Diagram

This sequence diagram depicts the execution lifecycle across the stages for a domain query and file mutation request

```mermaid
sequenceDiagram
    autonumber
    actor User as User Query
    participant Router as Semantic Query Router
    participant Engine as Agent Orchestrator
    participant Store as Hybrid Retrieval Store (Qdrant  BM25  Graph)
    participant Guard as Guardrails & Multi-Agent Defenses
    participant LLM as Local Ollama Runtime
    participant HITL as HITL
    participant Eval as Continuous Evaluator

    %% STAGE 1 & 2: INGESTION, CHUNKING & METADATA
    Note over User, Store: STAGES 1 & 2: UNIVERSAL INGESTION, CHUNKING & METADATA
    User->>Engine: Run indexing across data
    Engine->>Engine: Route File: PDF -> Docling | Image-Only PDF -> Tesseract OCR | Non-PDF -> Apache Tika
    Engine->>Engine: Structure via Unstructured & run Garbage-Text Safety Gate
    Engine->>Engine: Apply Heading Breadcrumb Lineage & Atomic Table Chunking
    Engine->>Engine: Precompute Summaries, TF-IDF Keywords & Hypothetical Questions
    Engine->>Store: Store BAAI/bge-m3 embeddings in Qdrant HNSW + BM25 index + NetworkX graph

    %% STAGE 4: ORCHESTRATION & ROUTING
    Note over User, Router: STAGE 4: QUERY TRIAGE & FAST-PASS ROUTING
    User->>Router: CLI Query: "Summarize GFS architecture and save to gfs.md"
    Router->>Router: Classify Query Triage Tier (DIRECT_MATH / STATUS vs VAULT_RAG)
    Router-->>Engine: Route to VAULT_RAG + FileAgent Pipeline

    %% STAGE 3: DATABASE STORAGE & 4-STAGE HYBRID RETRIEVAL FUNNEL
    Note over Engine, Store: STAGE 3: 4-STAGE HYBRID RETRIEVAL FUNNEL
    Engine->>Store: Native Metadata Filtering using Qdrant Filters
    Engine->>Store: Wide Hybrid RRF Fusion
    Engine->>Store: Bounded Graph Neighborhood Expansion 
    Engine->>Store: Precision Cross-Encoder Reranking

    %% STAGE 5: MULTI-AGENT ENGINE & HITL STATE MACHINE
    Note over Engine, HITL: STAGE 5: 16-AGENT LAYER 5 ENGINE & HITL GATE
    Engine->>Guard: Checkpoint: Audit inbound prompt & retrieved chunks against Indirect Injection
    Guard-->>Engine: Verified safe context blocks (<<< UNTRUSTED_DATA_BLOCK >>>)
    Engine->>LLM: Run Stage Sequence 
    LLM-->>Engine: Synthesized answer + File mutation action request
    Engine->>HITL: Emit PENDING request & persist MissionSnapshot
    Note over HITL, Engine: Filesystem write suspended until explicit user approval via CLI
    User->>HITL: Approve action
    HITL->>Engine: Authorize FileAgent atomic write

    %% STAGE 6: CONTINUOUS EVALUATION & REPORT CARD
    Note over Engine, Eval: STAGE 6: CONTINUOUS EVALUATION REPORT CARD
    Engine->>Eval: Evaluate Generation Triad & Cloud Cost Comparison
    Eval->>Eval: Compute N-gram Jaccard Faithfulness (S_faith) & Sigmoid Relevance
    Eval->>Eval: Estimate Production Cloud Pricing
    Eval-->>Engine: Return RAGReportCard & asynchronously log to .data/rag_eval_history.json
    Engine-->>User: Display Formatted Markdown Answer & Heuristic Report Card
```