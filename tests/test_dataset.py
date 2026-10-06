import csv
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_search_spanish import acquisition
from semantic_search_spanish.dataset import (
    SPLITS, SourceHeadline, corpus_statistics, load_corpus, load_gold, prepare_corpus,
)


class PreparationTests(unittest.TestCase):
    def records(self):
        return (
            SourceHeadline("t1", "train", 1, "  cafe\u0301\t azul "),
            SourceHeadline("t2", "train", 2, "Luna verde"),
            SourceHeadline("d1", "dev", 1, "café azul"),
            SourceHeadline("d2", "dev", 2, "  "),
            SourceHeadline("s1", "test", 1, "Luna verde"),
            SourceHeadline("s2", "test", 2, "Café azul"),
        )

    def test_duplicates_missing_and_provenance(self):
        corpus = prepare_corpus(self.records())
        self.assertEqual([d.id for d in corpus.documents], ["t1", "t2", "s2"])
        self.assertEqual(corpus.documents[0].text, "café azul")
        self.assertEqual(corpus.documents[0].original_text, self.records()[0].original_text)
        self.assertEqual([r.event_id for r in corpus.provenance["t1"]], ["t1", "d1"])
        stats = corpus.statistics
        self.assertEqual(stats["combined"]["missing_headlines"], 1)
        self.assertEqual(stats["combined"]["exact_duplicates"]["extra_rows"], 1)
        self.assertEqual(stats["combined"]["normalized_duplicates"]["extra_rows"], 2)
        self.assertEqual(stats["cross_split_exact_headlines"]["groups"], 1)
        self.assertEqual(stats["cross_split_normalized_headlines"]["groups"], 2)
        self.assertEqual(corpus, prepare_corpus(reversed(self.records())))

    def test_duplicate_ids_and_conflicts(self):
        records = [SourceHeadline("same", "train", 1, "Luna verde"),
                   SourceHeadline("same", "dev", 1, "Luna verde")]
        self.assertEqual(corpus_statistics(records)["combined"]["duplicate_event_ids"]["extra_rows"], 1)
        corpus = prepare_corpus(records)
        self.assertEqual(len(corpus.documents), 1)
        self.assertEqual(len(corpus.provenance["same"]), 2)
        for text in ("Mar rojo", " "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prepare_corpus([records[0], SourceHeadline("same", "dev", 1, text)])

    def test_invalid_provenance_and_empty_input(self):
        self.assertEqual(prepare_corpus([]).documents, ())
        for record in (SourceHeadline("a", "silver", 1, "Luna verde"),
                       SourceHeadline("", "train", 1, "Luna verde")):
            with self.assertRaises(ValueError):
                prepare_corpus([record])

    def test_gold_only_loader_and_no_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            before = {}
            for split in SPLITS:
                path = directory / f"gold_{split}.csv"
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["event_id", "canonical_headline", "esg_E"])
                    writer.writerow([split, "Luna verde", "1"])
                before[path] = path.read_bytes()
            (directory / "silver.csv").write_text("invalid", encoding="utf-8")
            corpus = load_corpus(directory)
            self.assertEqual(len(corpus.documents), 1)
            self.assertEqual({r.split for r in corpus.provenance["train"]}, set(SPLITS))
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertFalse(hasattr(corpus.documents[0], "esg_E"))

    def test_loader_rejects_bad_schema_rows_and_ids(self):
        for content in ("title\nLuna verde\n", "event_id,canonical_headline\na\n",
                        "event_id,canonical_headline\n,Luna verde\n",
                        "event_id,canonical_headline\n a,Luna verde\n",
                        "event_id,event_id,canonical_headline\na,b,Luna verde\n"):
            with tempfile.TemporaryDirectory() as folder:
                Path(folder, "gold_train.csv").write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_gold(folder)


class AcquisitionTests(unittest.TestCase):
    def test_archival_destination_rejected_before_io(self):
        with patch.object(acquisition.urllib.request, "urlopen") as network:
            with self.assertRaises(ValueError):
                acquisition.download_gold(Path("_originals") / "nested")
            network.assert_not_called()

    def test_pinned_download_integrity_and_reuse_offline(self):
        payload = b"event_id,canonical_headline\na,Luna verde\n"
        hashes = {f"gold_{s}.csv": hashlib.sha256(payload).hexdigest() for s in SPLITS}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(acquisition, "EXPECTED_SHA256", hashes), patch.object(
                    acquisition.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(payload)) as network:
                paths = acquisition.download_gold(folder)
                self.assertEqual(network.call_count, 3)
                for call in network.call_args_list:
                    self.assertIn(f"/resolve/{acquisition.REVISION}/gold_", call.args[0])
                network.reset_mock()
                self.assertEqual(acquisition.download_gold(folder), paths)
                network.assert_not_called()
                paths[0].write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    acquisition.download_gold(folder)
                self.assertEqual(paths[0].read_bytes(), b"corrupt")
                network.assert_not_called()

    def test_failed_integrity_leaves_no_download(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(acquisition.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(ValueError):
                    acquisition.download_gold(folder)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
