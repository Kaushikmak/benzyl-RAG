"""Retriever-agnostic core interfaces for RAGShield."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """Universal representation of a retrieved document or chunk candidate."""
    doc_id: str
    content: str
    source: str
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: Optional[str] = None
    reputation_score: float = 1.0
    quarantined: bool = False
    quarantine_reason: Optional[str] = None

    @property
    def hybrid_score(self) -> float:
        return self.scores.get("hybrid", 0.0)


class BaseRetriever(ABC):
    """Abstract interface for retriever implementations."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Candidate]:
        """Retrieve candidates matching the given query."""
        pass


class DefenseStage(ABC):
    """Abstract interface for RAGShield defense filters or detectors."""

    @abstractmethod
    def filter_candidates(
        self,
        candidates: List[Candidate],
        query: str,
        **kwargs
    ) -> List[Candidate]:
        """Apply defense logic to inspect, score, filter, or quarantine candidates."""
        pass
