import json
import logging
import os
from datetime import datetime
from typing import List

import numpy as np

import app.config as config

logger = logging.getLogger(__name__)

AUTO_LINKS_PATH = "./.data/auto_links.json"


def _cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def auto_link_external_documents(graph, chunks: List, embedding_model) -> list[dict]:
    note_chunks = [c for c in chunks if c.metadata.get("doc_type") == "note"]
    external_chunks = [c for c in chunks if c.metadata.get("doc_type") in ("external", "file")]

    if not note_chunks or not external_chunks:
        return []

    note_names = sorted({c.metadata.get("note_name") for c in note_chunks})
    note_embeddings = embedding_model.embed_documents(note_names)
    note_map = {name: emb for name, emb in zip(note_names, note_embeddings)}

    grouped_external = {}
    for chunk in external_chunks:
        key = chunk.metadata.get("note_name")
        grouped_external.setdefault(key, []).append(chunk.page_content)

    auto_links = []
    for ext_name, texts in grouped_external.items():
        joined = "\n".join(texts)[:3000]
        ext_embedding = embedding_model.embed_query(joined)

        scored = []
        for note_name, note_emb in note_map.items():
            scored.append((note_name, _cosine(ext_embedding, note_emb)))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_links = scored[: config.AUTO_LINK_TOP_K]
        ext_node = ext_name
        graph.add_node(ext_node)

        for note_name, score in top_links:
            if score < config.AUTO_LINK_THRESHOLD:
                continue
            graph.add_edge(ext_node, note_name)
            auto_links.append(
                {
                    "external_node": ext_node,
                    "note_node": note_name,
                    "score": round(score, 4),
                    "method": "embedding_cosine",
                    "timestamp": datetime.now().isoformat(),
                    "auto_applied": True,
                }
            )

    os.makedirs(os.path.dirname(AUTO_LINKS_PATH), exist_ok=True)
    with open(AUTO_LINKS_PATH, "w", encoding="utf-8") as f:
        json.dump(auto_links, f, indent=2)

    logger.info("Auto-linked %s external-note edges", len(auto_links))
    return auto_links
