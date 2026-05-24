from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ScoredDoc:
    content: str
    doc: Any
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    feedback_score: float = 0.0
    rerank_score: float = 0.0
    source: str = ""
    doc_id: str = ""
    source_kind: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hybrid_score(self):
        return (
            self.vector_score
            + self.bm25_score
            + self.graph_score
            + self.feedback_score
        )
