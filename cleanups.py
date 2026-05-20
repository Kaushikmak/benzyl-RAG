from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import TextLoader
import os
import re
import networkx as nx

# dir. path
OBSIDIAN_VAULT = "./obsidianVault"

loader = DirectoryLoader(
    OBSIDIAN_VAULT,
    glob="**/*.md",
    loader_cls=TextLoader,
    show_progress=True
)

documents = loader.load()

print("Building Obsidian graph...")

graph = nx.Graph()

note_map = {}
note_map_lower = {}  # Lowercase mapping for case-insensitive lookup

for doc in documents:
    source = doc.metadata["source"]

    filename = os.path.basename(source)

    note_name = filename.replace(".md", "")

    note_map[note_name] = doc
    note_map_lower[note_name.lower()] = note_name  # Map lowercase to original

    graph.add_node(note_name)

# internal link
wikilink_pattern = r"\[\[(.*?)\]\]"

for note_name, doc in note_map.items():
    links = re.findall(
        wikilink_pattern,
        doc.page_content
    )

    for link in links:
        clean_link = link.split("|")[0]
        
        # Case-insensitive lookup
        normalized_link = note_map_lower.get(clean_link.lower())
        
        if normalized_link:
            graph.add_edge(note_name, normalized_link)


print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print("Creating BM25 index...")
chunk_texts = [doc.page_content for doc in chunks]
tokenized_chunks = [
    text.lower().split()
    for text in chunk_texts
]

bm25 = BM25Okapi(tokenized_chunks)
with open("graph.pkl", "wb") as f:
    pickle.dump(graph, f)

with open("bm25.pkl", "wb") as f:
    pickle.dump((bm25, chunks), f)

print(f"Created {len(chunks)} chunks")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# Vector DB
db = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="./vectorstore"
)
print("Vector database created")
# Node DB
node_names = list(graph.nodes)
node_db = Chroma.from_texts(
    texts=node_names,
    embedding=embedding_model,
    persist_directory="./node_vectorstore"
)

print("Node database created")