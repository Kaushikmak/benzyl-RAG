import os
import hashlib
import numpy as np
from typing import Dict, List
from app.data import ScoredDoc


def _doc_id(doc) -> str:
    src = doc.metadata.get("source", "")
    payload = f"{src}|{doc.page_content[:500]}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def retrieve_vector(db, normalize_scores, query: str, k: int = 6) -> List[ScoredDoc]:
    results = db.similarity_search_with_relevance_scores(query, k=k)
    scores = [score for _, score in results]
    normalized = normalize_scores(scores)
    scored_docs = []

    for (doc, _), norm_score in zip(results, normalized):
        scored_docs.append(
            ScoredDoc(
                content=doc.page_content,
                doc=doc,
                vector_score=norm_score,
                source=os.path.basename(doc.metadata.get("source", "Unknown")),
                doc_id=_doc_id(doc),
                metadata=dict(doc.metadata),
            )
        )
    return scored_docs


def retrieve_bm25(bm25, bm25_chunks, normalize_scores, query: str, k: int = 6) -> List[ScoredDoc]:
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:k]
    scores = [bm25_scores[idx] for idx in top_indices]
    normalized = normalize_scores(scores)
    scored_docs = []

    for idx, norm_score in zip(top_indices, normalized):
        doc = bm25_chunks[idx]
        scored_docs.append(
            ScoredDoc(
                content=doc.page_content,
                doc=doc,
                bm25_score=norm_score,
                source=os.path.basename(doc.metadata.get("source", "Unknown")),
                doc_id=_doc_id(doc),
                metadata=dict(doc.metadata),
            )
        )

    return scored_docs


def merge_results(
    vector_docs: List[ScoredDoc],
    bm25_docs: List[ScoredDoc],
    weights: Dict[str, float],
    feedback_priors: Dict[str, float] | None = None,
) -> List[ScoredDoc]:
    merged = {}
    feedback_priors = feedback_priors or {}

    for doc in vector_docs + bm25_docs:
        if doc.doc_id not in merged:
            merged[doc.doc_id] = doc
        else:
            merged[doc.doc_id].vector_score += doc.vector_score
            merged[doc.doc_id].bm25_score += doc.bm25_score

    for doc in merged.values():
        prior = feedback_priors.get(doc.doc_id, 0.0)
        doc.feedback_score = prior * weights.get("feedback", 0.0)
        doc.vector_score = doc.vector_score * weights.get("vector", 1.0)
        doc.bm25_score = doc.bm25_score * weights.get("bm25", 1.0)

    return sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)
