from dataclasses import dataclass

@dataclass
class ScoredDoc:
    content: str
    doc: any
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""

    @property
    def hybrid_score(self):
        return self.vector_score + self.bm25_score