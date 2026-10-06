"""Explicit dense encoders and identity-checked retrieval. No import-time loads."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Sequence
from uuid import uuid4

import numpy as np

from .data import Document, normalize_text
from .retrieval import cosine_top_k, validate_documents, validate_k


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str
    pooling: str
    max_length: int = 128
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = True
    revision: str = "main"
    preprocessing: str = "nfc_whitespace"
    special_tokens: str = "included"

    def __post_init__(self):
        if self.pooling not in {"first_token", "masked_mean", "sentence_transformer"}:
            raise ValueError("Unsupported pooling")
        for value in (self.max_length, self.batch_size):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Lengths and batch sizes must be positive integers")
        if not self.model_name or not self.device or not self.revision:
            raise ValueError("Model, device and revision must be explicit")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be boolean")
        if self.preprocessing != "nfc_whitespace" or self.special_tokens != "included":
            raise ValueError("Only documented preprocessing and special-token policy are supported")

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EmbeddingBatch:
    identity: str
    owner: str
    vectors: np.ndarray


def pool_hidden_states(hidden, mask, pooling: str) -> np.ndarray:
    """CPU pooling of [batch, tokens, dimensions]; attended special tokens count."""
    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(mask)
    if (hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape
            or hidden.shape[1] == 0 or hidden.shape[2] == 0
            or not np.isfinite(hidden).all() or not np.isin(mask, [0, 1]).all()
            or (mask.sum(axis=1) == 0).any()):
        raise ValueError("Malformed hidden states or attention mask")
    if pooling == "first_token":
        if not (mask[:, 0] == 1).all():
            raise ValueError("First token must be attended (right padding required)")
        return hidden[:, 0].copy()
    if pooling == "masked_mean":
        return (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    raise ValueError("Unsupported manual pooling")


def _validated_vectors(vectors, rows: int, normalize: bool) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if (vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()):
        raise ValueError("Expected aligned, finite two-dimensional embeddings")
    if rows:
        scale = np.max(np.abs(vectors), axis=1, keepdims=True)
        if (scale == 0).any():
            raise ValueError("Zero embeddings are invalid for dense cosine retrieval")
        if normalize:
            vectors = vectors / scale
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.copy()
    vectors.setflags(write=False)
    return vectors


class _Encoder:
    @property
    def config(self):
        return self._config

    def _pack(self, vectors, rows):
        return EmbeddingBatch(self.config.identity, self._owner,
                              _validated_vectors(vectors, rows, self.config.normalize))

    def _texts(self, texts):
        if isinstance(texts, (str, bytes)):
            raise ValueError("Expected a sequence of texts")
        return [normalize_text(t) for t in texts]


class AutoModelEncoder(_Encoder):
    """One owned model/tokenizer pair; injected objects allow offline testing."""

    def __init__(self, config: EncoderConfig, *, model=None, tokenizer=None, torch_module=None,
                 local_files_only: bool = True):
        if config.pooling == "sentence_transformer":
            raise ValueError("Use SentenceEncoder for the library-defined pipeline")
        if (model is None) != (tokenizer is None):
            raise ValueError("Inject both model and tokenizer, or neither")
        if torch_module is None:
            import torch as torch_module
        if model is None:
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(config.model_name, revision=config.revision,
                                              local_files_only=local_files_only)
            resolved = getattr(model.config, "_commit_hash", None) or config.revision
            config = replace(config, revision=resolved)
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=resolved,
                                                      local_files_only=local_files_only)
        self._config, self._owner = config, uuid4().hex
        self._torch, self._model, self._tokenizer = torch_module, model, tokenizer
        self._model.to(config.device)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> EmbeddingBatch:
        texts = self._texts(texts)
        if not texts:
            return self._pack(np.empty((0, self._model.config.hidden_size)), 0)
        batches = []
        self._model.eval()
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                tokens = self._tokenizer(texts[start:start + self.config.batch_size],
                                         return_tensors="pt", padding=True, truncation=True,
                                         max_length=self.config.max_length)
                if "attention_mask" not in tokens or "input_ids" not in tokens:
                    raise ValueError("Tokenizer must supply input_ids and attention_mask")
                inputs = {key: value.to(self.config.device) for key, value in tokens.items()}
                hidden = self._model(**inputs).last_hidden_state.detach().cpu().numpy()
                mask = inputs["attention_mask"].detach().cpu().numpy()
                pooled = pool_hidden_states(hidden, mask, self.config.pooling)
                if pooled.shape[0] != len(texts[start:start + self.config.batch_size]):
                    raise ValueError("Model batch size does not match input")
                batches.append(pooled)
        return self._pack(np.concatenate(batches), len(texts))


class SentenceEncoder(_Encoder):
    """Use SentenceTransformer.encode and its pretrained pooling pipeline."""

    def __init__(self, config: EncoderConfig, *, model=None, local_files_only: bool = True):
        if config.pooling != "sentence_transformer":
            raise ValueError("SentenceEncoder requires library-defined pooling")
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.model_name, device=config.device,
                                        revision=config.revision, local_files_only=local_files_only)
            resolved = getattr(model[0].auto_model.config, "_commit_hash", None)
            if resolved:
                config = replace(config, revision=resolved)
        self._config, self._owner, self._model = config, uuid4().hex, model
        self._model.max_seq_length = config.max_length
        self._model.to(config.device)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> EmbeddingBatch:
        texts = self._texts(texts)
        if not texts:
            return self._pack(np.empty((0, self._model.get_sentence_embedding_dimension())), 0)
        self._model.eval()
        vectors = self._model.encode(texts, batch_size=self.config.batch_size,
                                     device=self.config.device, show_progress_bar=False,
                                     convert_to_numpy=True, normalize_embeddings=self.config.normalize)
        return self._pack(vectors, len(texts))


def dense_top_k(query: EmbeddingBatch, corpus: EmbeddingBatch,
                documents: Sequence[Document], *, k: int = 5):
    if query.identity != corpus.identity or query.owner != corpus.owner:
        raise ValueError("Query and corpus must come from the same encoder instance/configuration")
    q = _validated_vectors(query.vectors, 1, False)
    c = _validated_vectors(corpus.vectors, len(documents), False)
    return cosine_top_k(q[0], c, documents, k=k)


class DenseRetriever:
    """Encodes corpus and queries through one instance; accepts no external query vectors."""

    def __init__(self, documents: Sequence[Document], encoder):
        self.documents = tuple(documents)
        validate_documents(self.documents)
        self._encoder = encoder
        self._corpus = encoder.encode([d.text for d in self.documents])

    def search(self, query: str, *, k: int = 5):
        validate_k(k)
        return dense_top_k(self._encoder.encode([query]), self._corpus, self.documents, k=k)
