"""Judgment integrity only: no retrieval or metric calculation."""

import csv
import hashlib
import json
from pathlib import Path
import unittest
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts/human_review/candidates.csv"


class RelevanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.judgments = json.loads((ROOT / "evaluation/relevance.json").read_text(encoding="utf-8"))

    def test_complete_unique_query_ids(self):
        identifiers = [q["query_id"] for q in self.judgments]
        self.assertEqual(identifiers, [f"q{i:02d}" for i in range(1, 16)])
        self.assertEqual(len(set(identifiers)), 15)

    def test_positive_ids_unique_nonempty_and_valid(self):
        for query in self.judgments:
            with self.subTest(query_id=query["query_id"]):
                self.assertEqual(set(query), {"query_id", "relevant_event_ids"})
                ids = query["relevant_event_ids"]
                self.assertIsInstance(ids, list)
                self.assertGreater(len(ids), 0)
                self.assertEqual(len(ids), len(set(ids)))
                for identifier in ids:
                    self.assertEqual(str(UUID(identifier)), identifier)

    def test_frozen_queries_unchanged(self):
        digest = hashlib.sha256((ROOT / "evaluation/queries.json").read_bytes()).hexdigest()
        self.assertEqual(digest, "d1ede27ff1263a175864c34c6fb441bb5ef1685591475cf545cfd29f6f5c54e9")

    @unittest.skipUnless(POOL.exists(), "Blinded pool unavailable; membership not validated")
    def test_blinded_pool_unchanged_and_all_positives_present(self):
        self.assertEqual(hashlib.sha256(POOL.read_bytes()).hexdigest(),
                         "e6d1496267d8737ce12f1f88b587ed686def93a9d4f29d50625e0a75d5807fe0")
        with POOL.open(encoding="utf-8-sig", newline="") as handle:
            pairs = {(row["query_id"], row["event_id"]) for row in csv.DictReader(handle)}
        missing = [(q["query_id"], identifier) for q in self.judgments
                   for identifier in q["relevant_event_ids"]
                   if (q["query_id"], identifier) not in pairs]
        self.assertEqual(missing, [], "Supplied positives must belong to their query's pool")


if __name__ == "__main__":
    unittest.main()
