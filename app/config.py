VAULT_PATH = "/mnt/storage/Documents/Obsidian Vault/"

VECTORSTORE_DIR = "./vectorstore"
NODE_VECTORSTORE_DIR = "./node_vectorstore"

BM25_PATH = "./data/bm25.pkl"
GRAPH_PATH = "./data/graph.pkl"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

OLLAMA_MODEL = "qwen3:8b"

DEVICE = "cuda"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval knobs
VECTOR_K = 10
BM25_K = 10
MERGE_TOP_N = 12
GRAPH_TOP_MERGED = 4
GRAPH_MAX_NEIGHBORS = 2
GRAPH_MAX_CHUNKS_PER_NEIGHBOR = 2
RERANK_TOP_K = 4

# Weighted fusion
W_VECTOR = 1.0
W_BM25 = 1.0
W_GRAPH = 0.3
W_FEEDBACK = 0.8

# Feature toggles
ENABLE_QUERY_CACHE = True
ENABLE_WEB_AUGMENTATION = True
ENABLE_FEEDBACK_PRIORS = True

# Cache behavior
QUERY_CACHE_SIZE = 128

# Feedback behavior
FEEDBACK_DECAY = 0.95
FEEDBACK_MIN_VOTES = 2
FEEDBACK_MAX_BOOST = 0.25
FEEDBACK_MAX_PENALTY = -0.25
FEEDBACK_REFRESH_SECONDS = 60

# Web augmentation
WEB_SEARCH_TIMEOUT_SECONDS = 8
WEB_TOP_K = 3
TRUSTED_WEB_DOMAINS = [
    "docs.python.org",
    "fastapi.tiangolo.com",
    "streamlit.io",
    "platform.openai.com",
    "numpy.org",
    "pydantic.dev",
]

# Source preview
MAX_SOURCE_PREVIEW_CHARS = 30000

# External documents
EXTERNAL_DOCS_PATH = "./data/external_docs"
EXTERNAL_DOC_EXTENSIONS = [".pdf", ".docx", ".pptx"]
AUTO_LINK_TOP_K = 3
AUTO_LINK_THRESHOLD = 0.45
