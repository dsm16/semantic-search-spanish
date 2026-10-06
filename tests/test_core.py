import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_search_spanish import (
    clean_documents, cosine_top_k, load_csv, mean_reciprocal_rank,
    precision_at_k, reciprocal_rank,
)
from semantic_search_spanish.encoders import TfidfRetriever


class DataTests(unittest.TestCase):
    def test_normalization_and_stable_identity(self):
        docs = clean_documents(["  cafe\u0301\t azul ", "café azul", "Café azul"])
        self.assertEqual([d.id for d in docs], ["0", "2"])
        self.assertEqual(docs[0].text, "café azul")
        self.assertEqual(docs[0].original_text, "  cafe\u0301\t azul ")

    def test_optional_duplicates_and_empty_collection(self):
        self.assertEqual(len(clean_documents(["uno", "uno"], deduplicate=False)), 2)
        self.assertEqual(clean_documents([]), [])

    def test_invalid_inputs(self):
        for texts in ([None], [float("nan")], [" \t"], "headline"):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                clean_documents(texts)
        for ids in (["a"], ["a", "a"], ["a", " "]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                clean_documents(["uno", "dos"], ids=ids)

    def test_csv_validation_and_read_only_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "toy.csv"
            original = b"id,title\na,uno\nb,uno\nc,dos\n"
            path.write_bytes(original)
            self.assertEqual([d.id for d in load_csv(path, id_column="id")], ["a", "c"])
            self.assertEqual(path.read_bytes(), original)
            for content in ("other\nuno\n", "title,title\nuno,dos\n",
                            "title,extra\nuno\n", "title\nuno,dos\n", "title\n \n"):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_csv(path)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = clean_documents(["uno", "dos", "tres", "cuatro"])
        self.vectors = [[1, 0], [0, 1], [-1, 0], [2, 0]]

    def test_ranking_ties_and_structured_results(self):
        results = cosine_top_k([1, 0], self.vectors, self.docs, k=10)
        self.assertEqual([r.document_id for r in results], ["0", "3", "1", "2"])
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])
        self.assertEqual([r.score for r in results], [1, 1, 0, -1])
        self.assertEqual(results[0].text, "uno")
        self.assertEqual(len(cosine_top_k([1, 0], self.vectors, self.docs, k=2)), 2)

    def test_empty_corpus(self):
        self.assertEqual(cosine_top_k([1, 0], [], [], k=1), [])

    def test_invalid_vectors_and_alignment(self):
        for query in ([], [0, 0], [math.nan, 0], [math.inf, 0], [[1, 0]], [1]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                cosine_top_k(query, self.vectors, self.docs)
        for vectors in ([[1, 0]], [[0, 0]] * 4, [[math.nan, 0]] * 4):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                cosine_top_k([1, 0], vectors, self.docs)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                cosine_top_k([1, 0], self.vectors, self.docs, k=k)
        with self.assertRaises(ValueError):
            cosine_top_k([1, 0], [[1, 0]] * 2, [self.docs[0]] * 2)

    def test_large_finite_vectors(self):
        result = cosine_top_k([1e308, 1e308], [[1e308, 1e308]], self.docs[:1])
        self.assertAlmostEqual(result[0].score, 1)


class EvaluationTests(unittest.TestCase):
    def test_precision_and_missing_ranks(self):
        self.assertEqual(precision_at_k(["a", "b", "c"], {"b", "c"}, 2), 0.5)
        self.assertEqual(precision_at_k(["b"], {"b"}, 5), 0.2)
        self.assertEqual(precision_at_k([], set(), 5), 0)

    def test_rr_cutoff_and_mrr(self):
        self.assertEqual(reciprocal_rank(["a", "b"], {"b"}), 0.5)
        self.assertEqual(reciprocal_rank(["a", "b"], {"b"}, k=1), 0)
        self.assertEqual(reciprocal_rank(["a"], set()), 0)
        self.assertEqual(mean_reciprocal_rank(
            {"q1": ["a", "b"], "q2": ["c"]},
            {"q1": {"b"}, "q2": {"c"}}), 0.75)

    def test_invalid_judgments_and_rankings(self):
        for ranked, relevant in ((["a", "a"], {"a"}), (["a"], ["a", "a"]),
                                 ([""], set()), ("a", {"a"})):
            with self.subTest(ranked=ranked), self.assertRaises(ValueError):
                precision_at_k(ranked, relevant, 2)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                precision_at_k([], set(), k)
            with self.assertRaises(ValueError):
                reciprocal_rank([], set(), k=k)
        for rankings, relevance in (({}, {}), ({"q": []}, {})):
            with self.assertRaises(ValueError):
                mean_reciprocal_rank(rankings, relevance)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn unavailable; not installed")
class TfidfTests(unittest.TestCase):
    def test_retrieval_and_corpus_vocabulary(self):
        retriever = TfidfRetriever(clean_documents(["gato negro", "perro blanco"]))
        result = retriever.search("gato", k=1)
        self.assertEqual(result[0].document_id, "0")
        self.assertGreater(result[0].score, 0)
        before = dict(retriever._vectorizer.vocabulary_)
        unknown = retriever.search("astronauta")
        self.assertEqual([r.score for r in unknown], [0, 0])
        self.assertEqual(retriever._vectorizer.vocabulary_, before)

    def test_independent_instances(self):
        first = TfidfRetriever(clean_documents(["gato negro", "perro blanco"]))
        second = TfidfRetriever(clean_documents(["avión azul", "barco rojo"]))
        self.assertGreater(first.search("gato")[0].score, 0)
        self.assertEqual(second.search("gato")[0].score, 0)

    def test_invalid_corpus_query_and_k(self):
        for docs in ([], clean_documents(["a", "b"])):
            with self.assertRaises(ValueError):
                TfidfRetriever(docs)
        retriever = TfidfRetriever(clean_documents(["gato negro"]))
        with self.assertRaises(ValueError):
            retriever.search(" ")
        with self.assertRaises(ValueError):
            retriever.search("gato", k=0)


if __name__ == "__main__":
    unittest.main()
