"""Lexical retrieval; optional dense encoders live in dense.py."""

from typing import Sequence

from .data import Document, normalize_text
from .retrieval import SearchResult, rank_scores, validate_documents, validate_k


class TfidfRetriever:
    """Own a corpus-fitted vocabulary; queries only transform that vocabulary.

    Unigrams and scikit-learn's default token pattern are used. No stop-word
    list is assumed. Out-of-vocabulary queries return zero scores in corpus
    order, matching cosine's conventional zero-vector behavior in sklearn.
    """

    def __init__(self, documents: Sequence[Document]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.documents = tuple(documents)
        validate_documents(self.documents)
        if not self.documents:
            raise ValueError("TF-IDF requires a nonempty corpus")
        self._vectorizer = TfidfVectorizer(lowercase=True, norm="l2")
        self._matrix = self._vectorizer.fit_transform([d.text for d in self.documents])

    def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
        from sklearn.metrics.pairwise import cosine_similarity

        validate_k(k)
        query_vector = self._vectorizer.transform([normalize_text(query)])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        return rank_scores(self.documents, scores, k)
