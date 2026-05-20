from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import pickle
import os
import re
import networkx as nx
import logging
from collections import Counter


logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",)
logger = logging.getLogger(__name__)


OBSIDIAN_VAULT = "./obsidianVault"

logger.info("Loading documents...")
loader = DirectoryLoader(
    OBSIDIAN_VAULT,
    glob="**/*.md",
    loader_cls=TextLoader,
    show_progress=True
)
documents = loader.load()
logger.info(f"Loaded {len(documents)} documents")


logger.info("Building Obsidian graph...")
graph = nx.Graph()
note_map = {}
note_map_lower = {}

for doc in documents:
    source = doc.metadata["source"]
    filename = os.path.basename(source)
    note_name = filename.replace(".md", "")

    note_map[note_name] = doc
    note_map_lower[note_name.lower()] = note_name

    graph.add_node(note_name)

wikilink_pattern = r"\[\[(.*?)\]\]"

for note_name, doc in note_map.items():
    links = re.findall(wikilink_pattern, doc.page_content)
    for link in links:
        clean_link = link.split("|")[0].strip()
        normalized_link = note_map_lower.get(clean_link.lower())
        if normalized_link:
            graph.add_edge(note_name, normalized_link)

logger.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")


logger.info("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_documents(documents)

for chunk in chunks:
    source = chunk.metadata.get("source", "")
    filename = os.path.basename(source)
    note_name = filename.replace(".md", "")
    chunk.metadata["note_name"] = note_name
    chunk.metadata["filename"] = filename

logger.info(f"Created {len(chunks)} chunks")

# hashtable counter
chunks_per_note = Counter(chunk.metadata["note_name"] for chunk in chunks)

logger.info("Chunk distribution statistics:")
logger.info(f"Min chunks/note: {min(chunks_per_note.values())}")
logger.info(f"Max chunks/note: {max(chunks_per_note.values())}")
logger.info(f"Average chunks/note: "f"{sum(chunks_per_note.values()) / len(chunks_per_note):.2f}")


logger.info("Creating BM25 index...")
chunk_texts = [doc.page_content for doc in chunks]
def tokenize(text):
    text = text.lower()
    # fetch words
    # word-boundry [fetch word] word-boundry
    tokens = re.findall(r"\b\w+\b", text)
    return tokens

tokenized_chunks = [tokenize(text) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)
logger.info("BM25 index created")


logger.info("Saving graph index...")
with open("graph.pkl", "wb") as f:
    pickle.dump(graph, f)
logger.info("Graph saved to graph.pkl")


logger.info("Saving BM25 index...")
with open("bm25.pkl", "wb") as f:
    pickle.dump((bm25, chunks), f)
logger.info("BM25 index saved to bm25.pkl")


logger.info("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"}
)
logger.info("Embedding model loaded")

logger.info("Building chunk vector database...")
db = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="./vectorstore"
)
logger.info("Chunk vector database created")

logger.info("Building node vector database...")
node_names = list(graph.nodes)
node_db = Chroma.from_texts(
    texts=node_names,
    embedding=embedding_model,
    persist_directory="./node_vectorstore"
)
logger.info("Node vector database created")