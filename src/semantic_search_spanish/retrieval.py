"""Exact cosine retrieval with explicit validation and stable ties."""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .data import Document


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    text: str
    score: float
    rank: int


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def validate_documents(documents: Sequence[Document]) -> None:
    if any(not isinstance(d, Document) or not isinstance(d.id, str)
           or not d.id.strip() or not isinstance(d.text, str) or not d.text.strip()
           for d in documents):
        raise ValueError("Expected documents with nonblank IDs and text")
    if len({d.id for d in documents}) != len(documents):
        raise ValueError("Document IDs must be unique")


def rank_scores(documents: Sequence[Document], scores: Iterable[float],
                k: int) -> list[SearchResult]:
    validate_k(k)
    validate_documents(documents)
    scores = [float(s) for s in scores]
    if len(scores) != len(documents) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Scores must be finite and aligned with documents")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return [SearchResult(documents[i].id, documents[i].text, scores[i], rank)
            for rank, i in enumerate(order, 1)]


def _unit_vector(values: Iterable[float]) -> list[float]:
    try:
        vector = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected a one-dimensional numeric vector") from exc
    if not vector or not all(math.isfinite(v) for v in vector):
        raise ValueError("Vectors must be nonempty and finite")
    scale = max(abs(v) for v in vector)
    if scale == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    scaled = [v / scale for v in vector]
    norm = math.sqrt(math.fsum(v * v for v in scaled))
    return [v / norm for v in scaled]


def cosine_top_k(query: Iterable[float], vectors: Iterable[Iterable[float]],
                 documents: Sequence[Document], *, k: int = 5) -> list[SearchResult]:
    """Return ranked records; equal scores preserve corpus order.

    The caller must use the same representation space for query and documents.
    Zero/nonfinite vectors are rejected rather than assigned misleading scores.
    """
    validate_k(k)
    validate_documents(documents)
    query = _unit_vector(query)
    vectors = [_unit_vector(v) for v in vectors]
    if len(vectors) != len(documents):
        raise ValueError("Vector rows and documents must have equal lengths")
    if any(len(v) != len(query) for v in vectors):
        raise ValueError("Query and document dimensions must match")
    scores = [max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(query, v))))
              for v in vectors]
    return rank_scores(documents, scores, k)
