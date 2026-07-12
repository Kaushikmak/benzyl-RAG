"""Adversarial redteaming attack generators for RAG evaluation (Corpus Poisoning & Indirect Injection)."""

from guardrails.redteaming.corpus_poisoning import generate_poisoned_chunk, generate_near_duplicate_cluster
from guardrails.redteaming.indirect_injection import apply_injection_payload, INJECTION_TEMPLATES

__all__ = [
    "generate_poisoned_chunk",
    "generate_near_duplicate_cluster",
    "apply_injection_payload",
    "INJECTION_TEMPLATES",
]
