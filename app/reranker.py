from typing import List
from app.data import ScoredDoc

def rerank(reranker_model,query: str,docs: List[ScoredDoc],top_k: int = 3) -> List[ScoredDoc]:

    if not docs:
        return []

    pairs = [(query, doc.content) for doc in docs]
    try:
        scores = reranker_model.predict(pairs, batch_size=4)
    except Exception:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            scores = reranker_model.predict(pairs, batch_size=1)
        except Exception:
            if hasattr(reranker_model, "model"):
                reranker_model.model.to("cpu")
            scores = reranker_model.predict(pairs, batch_size=4)

    for doc, score in zip(docs, scores):
        doc.rerank_score = float(score)

    reranked = sorted(
        docs,
        key=lambda x: x.rerank_score,
        reverse=True
    )

    return reranked[:top_k]