"""Validate safe frozen presentation data without running retrieval."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PublicResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((ROOT / "evaluation/benchmark_results.json").read_text(encoding="utf-8"))

    def test_exact_systems_and_query_coverage(self):
        expected = ["tfidf", "beto_first_token", "beto_masked_mean",
                    "robertuito_first_token", "robertuito_masked_mean", "minilm"]
        self.assertEqual([s["system"] for s in self.results["systems"]], expected)
        for s in self.results["systems"]:
            self.assertEqual([q["query_id"] for q in s["per_query"]], [f"q{i:02d}" for i in range(1, 16)])
        meta = self.results["metadata"]
        self.assertEqual((meta["corpus_size"], meta["query_count"], meta["positive_judgments"], meta["judged_pairs"]), (1625, 15, 43, 648))

    def test_frozen_aggregate_values_and_input_hashes(self):
        expected = [(0.1467, 0.3889, 1.23), (0.0133, 0.0400, 58.77),
                    (0.2400, 0.4411, 52.49), (0.0800, 0.2169, 52.87),
                    (0.2933, 0.5833, 53.46), (0.2667, 0.5278, 14.24)]
        for system, values in zip(self.results["systems"], expected):
            self.assertEqual((round(system["precision_at_5"], 4), round(system["mrr_at_10"], 4), round(system["seconds"], 2)), values)
        for filename, key in [("queries.json", "query_set_sha256"), ("relevance.json", "relevance_sha256")]:
            self.assertEqual(hashlib.sha256((ROOT / "evaluation" / filename).read_bytes()).hexdigest(), self.results["metadata"][key])

    def test_safe_public_schema(self):
        self.assertEqual(set(self.results), {"metadata", "systems"})
        for s in self.results["systems"]:
            self.assertEqual(set(s), {"system", "configuration", "seconds", "per_query", "precision_at_5", "mrr_at_10"})
            for q in s["per_query"]:
                self.assertEqual(set(q), {"query_id", "precision_at_5", "reciprocal_rank_at_10"})
        text = json.dumps(self.results)
        for forbidden in ["_originals", "canonical_headline", "original_text", "source_hashes", "hits", "C:\\", "D:\\", "cache_folder"]:
            self.assertNotIn(forbidden, text)
