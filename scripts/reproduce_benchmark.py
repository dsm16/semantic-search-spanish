"""Optional future rerun at frozen revisions; never invoked by analysis notebook."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_search_spanish.acquisition import EXPECTED_SHA256, verify_file
from semantic_search_spanish.dataset import load_corpus
from semantic_search_spanish.dense import AutoModelEncoder, SentenceEncoder, EncoderConfig, DenseRetriever
from semantic_search_spanish.encoders import TfidfRetriever
from semantic_search_spanish.evaluation import precision_at_k, reciprocal_rank

def fingerprint(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/esg-spanish-events")
    parser.add_argument("--allow-model-downloads", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((ROOT / "evaluation/benchmark_results.json").read_text(encoding="utf-8"))
    meta = frozen["metadata"]
    for filename, key in [("queries.json", "query_set_sha256"), ("relevance.json", "relevance_sha256")]:
        verify_file(ROOT / "evaluation" / filename, meta[key])
    for filename, digest in EXPECTED_SHA256.items():
        verify_file(Path(args.data) / filename, digest)
    queries = json.loads((ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    relevance = {q["query_id"]: set(q["relevant_event_ids"]) for q in json.loads((ROOT / "evaluation/relevance.json").read_text(encoding="utf-8"))}
    corpus = load_corpus(args.data)
    if len(corpus.documents) != 1625:
        raise ValueError("Unexpected corpus size")
    payload = [{"event_id": d.id, "text": d.text, "original_text": d.original_text,
                "provenance": [asdict(r) for r in corpus.provenance[d.id]]} for d in corpus.documents]
    if hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != meta["corpus_identity"]:
        raise ValueError("Corpus identity changed")
    runs, pools = [], {q["query_id"]: set() for q in queries}
    for system in frozen["systems"]:
        start = time.perf_counter()
        if system["system"] == "tfidf":
            retriever = TfidfRetriever(corpus.documents)
        else:
            config = EncoderConfig(**system["configuration"])
            cls = SentenceEncoder if config.pooling == "sentence_transformer" else AutoModelEncoder
            retriever = DenseRetriever(corpus.documents, cls(config, local_files_only=not args.allow_model_downloads))
        rankings = {}
        for query in queries:
            ids = [r.document_id for r in retriever.search(query["query_text"], k=10)]
            rankings[query["query_id"]] = ids
            pools[query["query_id"]].update(ids)
        runs.append({"system": system["system"], "rankings": rankings, "seconds": time.perf_counter() - start})
        del retriever
        gc.collect()
    # Reject any changed union before interpreting absence of positives as zero.
    for qid, ids in pools.items():
        if fingerprint(ids) != meta["pool_fingerprints"][qid]:
            raise ValueError("Judged-pool coverage changed; fresh human review required: " + qid)
    for run in runs:
        run["per_query"] = [{"query_id": qid, "precision_at_5": precision_at_k(ids, relevance[qid], 5),
                            "reciprocal_rank_at_10": reciprocal_rank(ids, relevance[qid], k=10)}
                           for qid, ids in run.pop("rankings").items()]
        run["precision_at_5"] = sum(q["precision_at_5"] for q in run["per_query"]) / 15
        run["mrr_at_10"] = sum(q["reciprocal_rank_at_10"] for q in run["per_query"]) / 15
    target = ROOT / "artifacts/reproduction/results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print("Saved separate reproduction results:", target)

if __name__ == "__main__":
    main()
