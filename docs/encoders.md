# Retrieval system configurations

The experiment exposes exactly six systems in `systems.SYSTEMS`:

| System | Model | Pooling |
| --- | --- | --- |
| tfidf | Corpus-fitted scikit-learn TF-IDF | Not applicable |
| beto_first_token | dccuchile/bert-base-spanish-wwm-cased | Final hidden state at position 0 |
| beto_masked_mean | dccuchile/bert-base-spanish-wwm-cased | Attention-mask-aware final-layer mean |
| robertuito_first_token | pysentimiento/robertuito-base-cased | Final hidden state at position 0 |
| robertuito_masked_mean | pysentimiento/robertuito-base-cased | Attention-mask-aware final-layer mean |
| minilm | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Pretrained SentenceTransformer pipeline |

Dense defaults are explicit: CPU, batch size 32, maximum 128 tokens (including
special tokens), and L2-normalized CPU NumPy float32 embeddings. Cosine ranking
is shared across systems and ties preserve corpus order. TF-IDF uses unigrams,
lowercasing, the default scikit-learn token pattern, and L2 normalization; it
fits documents only and transforms queries without refitting.

All text passes through the existing NFC/whitespace cleaning, preserving case
and accents. Robertuito uses raw cleaned text without tweet-specific
preprocessing, as a controlled configuration; this differs from the tweet
preprocessing recommended by its model card and must be retained in experiment
documentation. No preprocessing variant is silently added.

## Rationale

First-token pooling reproduces the original academic representation as a
corrected historical baseline. It is not `pooler_output`. Masked mean is a
pooling ablation, not a presumed improvement: sum attended token vectors and
divide by the attended-token count. Attended special tokens are included;
padding tokens are excluded. BETO and Robertuito are general pretrained
encoders and are not claimed to be sentence-embedding models.

MiniLM supplies a representation designed for sentence similarity, following
the documented `SentenceTransformer.encode()` path rather than reconstructing
its pooling. Its documented maximum sequence length is 128. No model is
presumed superior before human evaluation. Raw cosine magnitudes across models
must not be used as comparative quality evidence.

Sources:
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://sbert.net/docs/package_reference/sentence_transformer/model.html
- https://huggingface.co/pysentimiento/robertuito-base-cased

## Ownership and identity

Each AutoModelEncoder owns its tokenizer/model pair and immutable EncoderConfig:
model name, requested/resolved revision, pooling, max length, normalization,
batch size, device, preprocessing, and special-token policy. Loading resolves
the model commit when available, then loads its tokenizer at that same revision.
SentenceEncoder owns one SentenceTransformer instance and its configuration.
Record the exposed configuration for future runs; `main` is a loading default,
not an immutable model pin. Supply a commit revision for strictly pinned runs.

Embedding batches carry a stable configuration hash plus an instance owner ID.
Dense retrieval rejects either configuration or owner mismatches, including
separate injected model instances with otherwise identical configurations.
DenseRetriever encodes documents and queries through the same instance and
accepts no external query vectors. Treat owned models/tokenizers as inference
objects: do not mutate weights, tokenizer settings, or internal pipeline after
constructing a retriever. Identity checks protect configuration mixups, not
deliberately forged batches or arbitrary in-place weight changes.

AutoModel inference explicitly calls eval(), uses torch.inference_mode(),
dynamic padding, truncation, attention masks and all returned tokenizer inputs.
Inputs and model move to the configured device. Final hidden states return to
CPU for pooling and storage. Mean pooling therefore does not retain GPU tensors.
SentenceTransformer handles batching/inference through its library path;
eval() is also explicit. Invalid/zero/nonfinite dense embeddings are rejected.

## Dependencies and execution

The lightweight package requires NumPy and scikit-learn. Optional dependencies
are declared in the `transformers` extra: torch, transformers and
sentence-transformers, with no GPU-specific pins. Nothing is installed by the
code. Importing modules loads no pretrained models.

`build_system(name, documents)` uses `local_files_only=True` by default. A
missing package or uncached model fails instead of silently downloading.
Constructing dense encoders or systems is an explicit inference action.
`local_files_only=False` is an explicit download opt-in. Real smoke tests
downloaded uncached models; the official benchmark reused local caches.
For custom batch/device/revision settings instantiate an EncoderConfig and
encoder directly, then pass it to DenseRetriever.

No embedding-cache framework or real-model smoke test was added. Routine tests
exercise batching, masks, pooling, inference/eval calls, identity checks,
normalization and retrieval using deterministic NumPy-backed fake objects.
Separate real smoke tests passed for BETO, Robertuito and MiniLM; these are
excluded from the routine unit suite.

The frozen queries and source materials remain unchanged. The official pooled benchmark and human judgments are documented in
benchmark.md and ../evaluation/README.md.
