"""Model-independent text retrieval core. No models are loaded on import."""

from .data import Document, clean_documents, load_csv
from .evaluation import mean_reciprocal_rank, precision_at_k, reciprocal_rank
from .retrieval import SearchResult, cosine_top_k

__all__ = [
    "Document", "SearchResult", "clean_documents", "load_csv", "cosine_top_k",
    "precision_at_k", "reciprocal_rank", "mean_reciprocal_rank",
]
