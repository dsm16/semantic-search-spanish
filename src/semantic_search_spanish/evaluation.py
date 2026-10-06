"""Binary-relevance metrics over unique document IDs, without inferred labels."""

from typing import Iterable, Mapping

from .retrieval import validate_k


def _ids(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Expected a collection of document IDs")
    values = list(values)
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Document IDs must be nonblank strings")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate document IDs are not valid evaluation input")
    return values


def precision_at_k(ranked_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float:
    """Relevant hits in the first k ranks / k; missing ranks count as misses."""
    validate_k(k)
    ranked, relevant = _ids(ranked_ids), set(_ids(relevant_ids))
    return sum(identifier in relevant for identifier in ranked[:k]) / k


def reciprocal_rank(ranked_ids: Iterable[str], relevant_ids: Iterable[str],
                    *, k: int | None = None) -> float:
    """1 / first relevant rank, or 0; optional cutoff yields RR@k."""
    if k is not None:
        validate_k(k)
    ranked, relevant = _ids(ranked_ids), set(_ids(relevant_ids))
    for rank, identifier in enumerate(ranked[:k] if k is not None else ranked, 1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: Mapping[str, Iterable[str]],
                         relevance: Mapping[str, Iterable[str]],
                         *, k: int | None = None) -> float:
    """Macro-average RR across exactly matching, nonempty query sets.

    Queries with no relevant documents contribute zero. Missing judgments
    raise an error instead of being silently treated as no relevance.
    """
    if not rankings or set(rankings) != set(relevance):
        raise ValueError("Rankings and relevance need identical nonempty query sets")
    return sum(reciprocal_rank(ranked, relevance[q], k=k)
               for q, ranked in rankings.items()) / len(rankings)
