import os

DATA_DIR = "./data"
STORAGE_DIR = "./.data"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

QDRANT_DIR = "./.data/qdrant_db"
QDRANT_CHUNK_COLLECTION = "document_chunks"
QDRANT_NODE_COLLECTION = "document_nodes"

VECTORSTORE_DIR = "./.data/qdrant_db"
NODE_VECTORSTORE_DIR = "./.data/qdrant_db"

BM25_PATH = "./.data/bm25.pkl"
GRAPH_PATH = "./.data/graph.pkl"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_NUM_CTX = 8192
OLLAMA_TIMEOUT = 180.0

DEVICE = "cuda"
EMBED_BATCH_SIZE = 4

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Retrieval knobs
VECTOR_K = 50
BM25_K = 50
MERGE_TOP_N = 40
GRAPH_TOP_MERGED = 3
GRAPH_MAX_NEIGHBORS = 2
GRAPH_MAX_CHUNKS_PER_NEIGHBOR = 2
RERANK_TOP_K = 8

# Weighted fusion
W_VECTOR = 1.0
W_BM25 = 1.0
W_GRAPH = 0.3
W_FEEDBACK = 0.8

# Feature toggles
ENABLE_QUERY_CACHE = True
ENABLE_FEEDBACK_PRIORS = True

# Cache behavior
QUERY_CACHE_SIZE = 128

# Feedback behavior
FEEDBACK_DECAY = 0.95
FEEDBACK_MIN_VOTES = 2
FEEDBACK_MAX_BOOST = 0.25
FEEDBACK_MAX_PENALTY = -0.25
FEEDBACK_MAX_STEP_PER_VOTE = 0.05
FEEDBACK_REFRESH_SECONDS = 60



# Source preview
MAX_SOURCE_PREVIEW_CHARS = 30000

# External documents
EXTERNAL_DOCS_PATH = DATA_DIR
EXTERNAL_DOC_EXTENSIONS = [
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".txt", ".csv",
    ".rtf", ".epub", ".odt", ".md", ".json", ".xml"
]
AUTO_LINK_TOP_K = 3
AUTO_LINK_THRESHOLD = 0.45

# Ingestion Pipeline Configuration
INGESTION_USE_TIKA = True
INGESTION_USE_DOCLING_FOR_PDF = True
INGESTION_USE_UNSTRUCTURED = True
INGESTION_UNIVERSAL_LOAD = True

# Intelligent Chunking & Metadata Extraction Configuration
ENABLE_INTELLIGENT_CHUNKING = True
PRESERVE_ATOMIC_TABLES = True
TABLE_MAX_CHARS = 4000
ENABLE_METADATA_EXTRACTION = True
USE_LLM_FOR_METADATA = False

# Orchestration & Query Router Configuration
ENABLE_QUERY_ROUTING = True
ROUTER_FAST_CLASSIFY = True

# Multi-Agent Execution & Safety Configuration
ENABLE_MULTI_AGENT_ORCHESTRATION = True
REQUIRE_HITL_FOR_MUTATIONS = True
MISSION_STATE_DIR = ".mission_state"

# Continuous Evaluation & Monitoring Configuration
ENABLE_CONTINUOUS_EVAL = True
EVAL_HISTORY_PATH = "./.data/rag_eval_history.json"
PRODUCTION_PRICING_USD_PER_M_TOKENS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "llama-3.3-70b-cloud": {"input": 0.60, "output": 0.60},
    "local-compute": {"input": 0.0, "output": 0.0},
}
