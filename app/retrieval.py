import os
import numpy as np
from typing import List
from app.data import ScoredDoc


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
                source=os.path.basename(
                    doc.metadata.get(
                        "source", "Unknown"
                    )
                )
            )
        )
    return scored_docs


def retrieve_bm25(bm25,bm25_chunks,normalize_scores,query: str,k: int = 6) -> List[ScoredDoc]:
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
                source=os.path.basename(
                    doc.metadata.get("source", "Unknown")
                )
            )
        )

    return scored_docs


def merge_results(vector_docs: List[ScoredDoc],bm25_docs: List[ScoredDoc]) -> List[ScoredDoc]:
    merged = {}

    for doc in vector_docs + bm25_docs:
        if doc.content not in merged:
            merged[doc.content] = doc
        else:
            merged[doc.content].vector_score += doc.vector_score
            merged[doc.content].bm25_score += doc.bm25_score

    return sorted(
        merged.values(),
        key=lambda x: x.hybrid_score,
        reverse=True
    )