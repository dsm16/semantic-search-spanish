"""Validate the frozen public query artifact without models or private data."""

import json
from pathlib import Path
import unittest


class FrozenQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "queries.json"
        cls.queries = json.loads(path.read_text(encoding="utf-8"))

    def test_exactly_fifteen_queries(self):
        self.assertIsInstance(self.queries, list)
        self.assertEqual(len(self.queries), 15)

    def test_only_required_fields(self):
        for query in self.queries:
            with self.subTest(query=query):
                self.assertEqual(set(query), {"query_id", "query_text"})

    def test_unique_complete_stable_ids(self):
        identifiers = [query["query_id"] for query in self.queries]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(identifiers, [f"q{i:02d}" for i in range(1, 16)])

    def test_nonblank_query_text(self):
        for query in self.queries:
            with self.subTest(query_id=query["query_id"]):
                self.assertIsInstance(query["query_text"], str)
                self.assertTrue(query["query_text"].strip())


if __name__ == "__main__":
    unittest.main()
