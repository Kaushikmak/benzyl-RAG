import os
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any
from app.data import ScoredDoc


def _doc_id(doc) -> str:
    src = doc.metadata.get("source", "")
    payload = f"{src}|{doc.page_content[:500]}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_qdrant_filter(filter_dict: Optional[Dict[str, Any]] = None):
    """Build native Qdrant models.Filter from metadata filter dictionary."""
    if not filter_dict:
        return None
    try:
        from qdrant_client.http import models as qmodels
        conditions = []
        for k, v in filter_dict.items():
            key = f"metadata.{k}"
            conditions.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=v)
                )
            )
        return qmodels.Filter(must=conditions)
    except Exception:
        return None


def retrieve_vector(
    db,
    normalize_scores,
    query: str,
    k: int = 50,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[ScoredDoc]:
    """Retrieve dense vector candidates with optional native Qdrant metadata pre-filtering."""
    q_filter = build_qdrant_filter(filter_dict)
    try:
        if q_filter is not None:
            results = db.similarity_search_with_relevance_scores(query, k=k, filter=q_filter)
        else:
            results = db.similarity_search_with_relevance_scores(query, k=k)
    except TypeError:
        # Fallback if vectorstore implementation doesn't accept filter kwarg
        results = db.similarity_search_with_relevance_scores(query, k=k)

    # Post-filter fallback if filter applied post-hoc or TypeError occurred
    if filter_dict:
        filtered_results = []
        for doc, score in results:
            if all(doc.metadata.get(key) == val for key, val in filter_dict.items()):
                filtered_results.append((doc, score))
        results = filtered_results[:k]

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


def retrieve_bm25(
    bm25,
    bm25_chunks,
    normalize_scores,
    query: str,
    k: int = 50,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[ScoredDoc]:
    """Retrieve exact keyword candidates with hard metadata pre-filtering."""
    tokenized_query = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype=float)

    if filter_dict:
        for idx, doc in enumerate(bm25_chunks):
            if not all(doc.metadata.get(key) == val for key, val in filter_dict.items()):
                bm25_scores[idx] = -float("inf")

    # Select top k indices with score > -inf
    valid_indices = [idx for idx in range(len(bm25_scores)) if bm25_scores[idx] > -float("inf")]
    if not valid_indices:
        return []

    top_indices = sorted(valid_indices, key=lambda idx: bm25_scores[idx], reverse=True)[:k]
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
    """Merge and rank candidates using Reciprocal Rank Fusion / Weighted Hybrid Score."""
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
