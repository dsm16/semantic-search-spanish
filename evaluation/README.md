# Frozen human evaluation queries

`queries.json` is the official UTF-8 query set: 15 Spanish information needs
with stable IDs `q01` through `q15`. Each record contains only `query_id` and
`query_text`. The text was approved by the human reviewer and is frozen before
comparative retrieval experiments.

Candidates were developed through transparent headline inspection and string
filtering of the deduplicated gold corpus, then selected and revised by the
human reviewer. No model outputs, similarity rankings, or comparative model
performance were used to select these queries.

## Human relevance judgments

`relevance.json` transcribes the human reviewer's supplied positive event IDs
without headlines or automatic reinterpretation. Judgments were produced
manually from the blinded pooled candidates: system identity, scores, ranks,
and retrieval provenance were hidden, and candidate order was shuffled.

**Validation status: frozen.** All 43 supplied positive judgments were checked
against their respective blinded query pools. The human reviewer explicitly
corrected a transcription error in q07 before freezing; no judgments were
inferred or reinterpreted by the implementation.

Listed query-document pairs have binary relevance 1. Every
other document in that query's blinded pool has binary relevance 0. Documents
outside that query's pool remain UNJUDGED, not validated negatives. The
positive-only JSON must always be interpreted with its associated pool;
absence from the positive list alone does not establish relevance 0.

The pool contains the union of each of six systems' top 10 results. The
official metrics are **Precision@5 and MRR@10**. Unrestricted MRR must not be
reported as though all 1,625 corpus documents were exhaustively judged. Before
evaluating any ranking, verify its evaluated positions are covered by the
judged pool. The small manually judged test set and pooling coverage limit
generalization. The official benchmark report is available in ../docs/benchmark.md.

Associated frozen inputs (SHA-256):

- queries.json: `d1ede27ff1263a175864c34c6fb441bb5ef1685591475cf545cfd29f6f5c54e9`
- blinded candidates.csv: `e6d1496267d8737ce12f1f88b587ed686def93a9d4f29d50625e0a75d5807fe0`

The blinded pool remains in ignored `artifacts/human_review/candidates.csv`.
Pool-membership tests run when that artifact is available; they explicitly
skip when it is absent, rather than implying membership validation succeeded.

Judge headline meaning against the query's information need. Shared companies,
keywords, ESG labels, sentiment, and other metadata do not establish relevance.
Documents outside the judged pool remain unjudged, rather than automatically
irrelevant. Pool construction and the judging rubric must be documented before
benchmark metrics are reported.

Do not revise queries in response to model performance. Any later substantive
change requires an explicitly documented new query-set version.

Published aggregate and per-query measurements are in `benchmark_results.json`;
this safe artifact contains no headlines, scores, rankings or internal pool
provenance. Pool fingerprints allow a future reproduction to verify judged
coverage without publishing the ignored pool.
