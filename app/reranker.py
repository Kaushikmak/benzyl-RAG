from typing import List
from app.data import ScoredDoc

def rerank(reranker_model,query: str,docs: List[ScoredDoc],top_k: int = 3) -> List[ScoredDoc]:

    if not docs:
        return []

    pairs = [(query, doc.content) for doc in docs]
    scores = reranker_model.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.rerank_score = float(score)

    reranked = sorted(
        docs,
        key=lambda x: x.rerank_score,
        reverse=True
    )

    return reranked[:top_k]