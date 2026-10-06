# Spanish headline retrieval: lexical and pretrained representations

A compact information-retrieval experiment comparing TF-IDF, BETO, Robertuito
and a sentence-embedding model over Spanish headlines. The focus is correctness,
explicit encoder ownership, and a human evaluation defined before seeing model
performance—not a deployed search product.

## Why rebuild the academic exercise?

The original assignment compared BETO and Robertuito using first-token vectors.
An audit found that the visible flow overwrote global model/tokenizer variables
with Robertuito before generating queries labeled BETO. A sequential run would
compare query and document vectors from different encoders, even though both
have 768 dimensions. Incomplete saved execution history leaves historical
outputs ambiguous; the original superiority claim is not reused as evidence.

The rebuilt implementation binds corpus and query encoding to one encoder
instance/configuration and rejects identity mismatches. The academic notebook
and professor-provided data remain excluded from the public project.

## Experiment

```text
Pinned gold CSVs → 1,625 deduplicated headlines → six frozen systems
15 queries frozen before model runs → top 10 per system
→ union and shuffle → blinded human review → Precision@5 and MRR@10
```

The gold train/dev/test files contain 1,688 records. NFC/whitespace cleaning and
exact normalized-title deduplication produce 1,625 documents with stable
representative event IDs and complete source provenance. Classification splits
are combined as a retrieval corpus; their ESG labels are never relevance labels.

Six systems: TF-IDF; BETO first-token and masked mean; Robertuito first-token
and masked mean; `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
First-token is the corrected historical baseline. Mean pooling includes
attended special tokens and excludes padding. MiniLM uses the documented
SentenceTransformer pipeline. Dense settings: CPU, batch size 32, 128-token
maximum, L2 normalization. See [encoder details](docs/encoders.md).

The reviewer judged 648 unique query-document pairs while system identity,
scores and ranks were hidden. There are 43 positives; other in-pool documents
are binary 0 and outside-pool documents remain unjudged.

## Frozen benchmark

| System | Precision@5 | MRR@10 | Runtime (s) |
| --- | ---: | ---: | ---: |
| TF-IDF | 0.1467 | 0.3889 | 1.23 |
| BETO first-token | 0.0133 | 0.0400 | 58.77 |
| BETO masked mean | 0.2400 | 0.4411 | 52.49 |
| Robertuito first-token | 0.0800 | 0.2169 | 52.87 |
| Robertuito masked mean | 0.2933 | 0.5833 | 53.46 |
| MiniLM | 0.2667 | 0.5278 | 14.24 |

Robertuito masked mean has the highest aggregate metrics **on this test set**. MiniLM is close in quality and took 14.24 s versus 53.46 s in this CPU run. Masked mean substantially exceeds first-token pooling within both original encoders; TF-IDF remains a fast lexical baseline. Cross-model differences cannot be attributed solely to pooling: training data, architecture details and tokenization also differ. Raw cosine magnitudes are not compared, and no statistical significance or universal superiority is claimed.

[Analysis notebook](notebooks/01_semantic_search_benchmark.ipynb) ·
[Full report and per-query table](docs/benchmark.md) ·
[Public results JSON](evaluation/benchmark_results.json)

![MRR at 10](docs/figures/mrr_at_10.png)

## Repository

```text
evaluation/    frozen queries, positive judgments, safe results
src/semantic_search_spanish/    validation, encoders, retrieval, metrics
notebooks/     analysis of published measurements only
docs/          methodology, results, figures and attribution
scripts/       figure rendering and optional benchmark reproduction
tests/         offline deterministic tests
```

Raw data, model caches, internal rankings and runtime artifacts are ignored.
The public notebook never accesses archival files, downloads models or runs
retrieval; it reads the safe result JSON and imports the reusable plot helper.

## Installation and tests

Python 3.10+; no GPU-specific dependency pins. Choose the dependencies needed:

Installation requires internet unless dependencies are already available from
a local package cache. Dataset acquisition and uncached model downloads also
require internet. Tests and analysis of the committed results run offline.

```powershell
python -m pip install -e .
python -m pip install -e ".[transformers]"  # optional model execution
python -m pip install -e ".[analysis]"      # optional notebook/figures
python -B -m unittest discover -s tests -v
```

Tests use toy text/vectors and mocked models. They need no internet or model
downloads. Pool-membership checks explicitly skip if the ignored local pool is
absent; this is not a claim that missing data were validated.

## Dataset acquisition and reproduction

[DReggio/esg-spanish-events](https://huggingface.co/datasets/DReggio/esg-spanish-events)
indicates **CC BY-NC 4.0**. Attribution and source must be preserved; code and
external data are separate works. This project does not claim ownership of the
dataset or assign it a new license. Project code is licensed under
[MIT](LICENSE). The MIT license does not relicense third-party data; the
external dataset retains its own CC BY-NC 4.0 terms.

```powershell
$env:PYTHONPATH = 'src'
python -B -m semantic_search_spanish.acquisition
python -B scripts/render_figures.py  # reads frozen results; no inference
```

Acquisition downloads only the three gold files at the pinned commit and
verifies SHA-256; destination is ignored `data/raw/esg-spanish-events/`.
No Hugging Face account or archival directory is required.

To **optionally rerun** the frozen experiment on another machine:

```powershell
python -B scripts/reproduce_benchmark.py --allow-model-downloads
```

This uses the published model commit revisions and existing preparation code.
Without the download flag, models must already be cached. It checks input hashes,
corpus identity and query-specific pooled-ID fingerprints before scoring. If
pool coverage changes, it stops rather than assigning unjudged negatives.
Rerun results go to ignored `artifacts/reproduction/` and never overwrite the
official artifact. The reproduction script was added without executing it.
Exact software versions are recorded in the public JSON; different platforms
or versions may affect rankings and runtime.

## Limits

The evaluation contains only 15 manually selected information needs and 43 positive judgments, covering 648 pooled query-document pairs. Queries were informed by corpus inspection, but selected before model outputs. Judgments use headline meaning, not ESG metadata. Documents outside each query pool remain unjudged. MRR is restricted to 10; full-corpus recall and unrestricted MRR are unsupported. Some queries have fewer than five positives, limiting their attainable Precision@5. Human ambiguity, title deduplication, truncation to 128 tokens, and the absence of tweet-specific Robertuito preprocessing constrain interpretation. Runtime includes model construction/loading, corpus encoding, query retrieval and scoring; cached CPU wall time is not a controlled latency benchmark.

See [dataset documentation](docs/dataset.md) and [judging protocol](evaluation/README.md).
