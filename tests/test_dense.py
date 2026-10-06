from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_search_spanish.data import clean_documents
from semantic_search_spanish.dense import (
    AutoModelEncoder, DenseRetriever, EncoderConfig, SentenceEncoder,
    dense_top_k, pool_hidden_states,
)
from semantic_search_spanish.systems import SYSTEMS


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def detach(self):
        return self

    def cpu(self):
        self.device = "cpu"
        return self

    def numpy(self):
        return self.values


class FakeTorch:
    active = False

    @contextmanager
    def inference_mode(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": Tensor([[1, 2, 0]] * len(texts)),
                "attention_mask": Tensor([[1, 1, 0]] * len(texts)),
                "token_type_ids": Tensor([[0, 0, 0]] * len(texts))}


class Model:
    config = SimpleNamespace(hidden_size=2)

    def __init__(self, torch):
        self.torch, self.training, self.calls = torch, True, 0

    def to(self, device):
        self.device = device

    def eval(self):
        self.training = False

    def __call__(self, **inputs):
        assert not self.training and self.torch.active
        assert all(t.device == self.device for t in inputs.values())
        assert "attention_mask" in inputs and "token_type_ids" in inputs
        self.calls += 1
        return SimpleNamespace(last_hidden_state=Tensor(
            [[[3, 0], [0, 4], [900, 900]]] * len(inputs["input_ids"].values)))


class SentenceModel:
    def __init__(self):
        self.calls = []
        self.training = True

    def to(self, device):
        self.device = device

    def eval(self):
        self.training = False

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        assert not self.training
        self.calls.append((texts, kwargs))
        return np.array([[3, 4]] * len(texts))


class DenseTests(unittest.TestCase):
    def encoder(self, pooling="masked_mean", normalize=True):
        config = EncoderConfig("fake", pooling, batch_size=2, normalize=normalize)
        torch = FakeTorch()
        model, tokenizer = Model(torch), Tokenizer()
        return AutoModelEncoder(config, model=model, tokenizer=tokenizer, torch_module=torch)

    def test_masked_mean_and_first_token(self):
        hidden = [[[3, 0], [0, 4], [900, 900]]]
        np.testing.assert_allclose(pool_hidden_states(hidden, [[1, 1, 0]], "masked_mean"), [[1.5, 2]])
        np.testing.assert_allclose(pool_hidden_states(hidden, [[1, 1, 0]], "first_token"), [[3, 0]])

    def test_batching_device_inference_and_normalization(self):
        encoder = self.encoder()
        batch = encoder.encode(["uno", "dos", "tres", "cuatro", "cinco"])
        self.assertEqual(encoder._model.calls, 3)
        self.assertEqual([len(c[0]) for c in encoder._tokenizer.calls], [2, 2, 1])
        self.assertEqual(encoder._tokenizer.calls[0][1],
                         dict(return_tensors="pt", padding=True, truncation=True, max_length=128))
        np.testing.assert_allclose(np.linalg.norm(batch.vectors, axis=1), 1)
        self.assertFalse(batch.vectors.flags.writeable)
        self.assertFalse(encoder._model.training)
        self.assertFalse(encoder._torch.active)
        np.testing.assert_allclose(self.encoder(normalize=False).encode(["uno"]).vectors, [[1.5, 2]])
        np.testing.assert_allclose(self.encoder("first_token").encode(["uno"]).vectors, [[1, 0]])

    def test_empty_and_invalid_text(self):
        encoder = self.encoder()
        self.assertEqual(encoder.encode([]).vectors.shape, (0, 2))
        self.assertEqual(encoder._model.calls, 0)
        for texts in ([" "], [None], "uno"):
            with self.assertRaises(ValueError):
                encoder.encode(texts)

    def test_malformed_pooling_tensors(self):
        for hidden, mask in (([[1, 2]], [[1]]), ([[[1, 2]]], [[0]]),
                             ([[[np.nan, 2]]], [[1]]), ([[[1, 2]]], [[2]]),
                             ([[[1, 2]]], [[1, 1]])):
            with self.assertRaises(ValueError):
                pool_hidden_states(hidden, mask, "masked_mean")
        with self.assertRaises(ValueError):
            pool_hidden_states([[[1, 2], [3, 4]]], [[0, 1]], "first_token")

    def test_identity_record_and_mismatch_rejection(self):
        encoder = self.encoder()
        docs = clean_documents(["uno"])
        corpus = encoder.encode(["uno"])
        self.assertEqual(dense_top_k(encoder.encode(["dos"]), corpus, docs)[0].document_id, "0")
        for field, value in (("model_name", "other"), ("pooling", "first_token"),
                             ("max_length", 64), ("normalize", False), ("revision", "abc")):
            config = replace(encoder.config, **{field: value})
            self.assertNotEqual(config.identity, encoder.config.identity)
            with self.assertRaises(ValueError):
                dense_top_k(replace(corpus, identity=config.identity), corpus, docs)
        other = self.encoder()
        self.assertEqual(other.config.identity, encoder.config.identity)
        with self.assertRaises(ValueError):
            dense_top_k(other.encode(["uno"]), corpus, docs)

    def test_identity_change_after_corpus_rejected(self):
        encoder = self.encoder()
        retriever = DenseRetriever(clean_documents(["uno"]), encoder)
        encoder._config = replace(encoder.config, max_length=64)
        with self.assertRaises(ValueError):
            retriever.search("uno")

    def test_retrieval_ties_and_empty_corpus(self):
        retriever = DenseRetriever(clean_documents(["uno", "dos"]), self.encoder())
        self.assertEqual([r.document_id for r in retriever.search("tres")], ["0", "1"])
        self.assertEqual(len(retriever.search("tres", k=1)), 1)
        self.assertEqual(DenseRetriever([], self.encoder()).search("uno"), [])

    def test_bad_model_outputs_and_tokenizer(self):
        encoder = self.encoder()
        class BadModel(Model):
            def __call__(self, **inputs):
                return SimpleNamespace(last_hidden_state=Tensor([[[0, 0], [0, 0], [0, 0]]]))
        encoder._model = BadModel(encoder._torch)
        with self.assertRaises(ValueError):
            encoder.encode(["uno"])
        encoder._tokenizer = lambda *a, **k: {"input_ids": Tensor([[1]])}
        with self.assertRaises(ValueError):
            encoder.encode(["uno"])

    def test_sentence_library_path(self):
        model = SentenceModel()
        config = EncoderConfig("fake-sentence", "sentence_transformer", batch_size=2)
        encoder = SentenceEncoder(config, model=model)
        batch = encoder.encode(["uno", "dos"])
        np.testing.assert_allclose(batch.vectors, [[0.6, 0.8]] * 2)
        self.assertEqual(model.max_seq_length, 128)
        self.assertEqual(model.calls[0][1], dict(batch_size=2, device="cpu",
                         show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True))
        self.assertEqual(encoder.encode([]).vectors.shape, (0, 2))
        with self.assertRaises(ValueError):
            SentenceEncoder(EncoderConfig("fake", "masked_mean"), model=model)

    def test_config_validation_and_exact_six_systems(self):
        self.assertEqual(len(SYSTEMS), 6)
        for kwargs in ({"batch_size": 0}, {"max_length": True}, {"pooling": "other"}):
            values = {"model_name": "fake", "pooling": "first_token", **kwargs}
            with self.assertRaises(ValueError):
                EncoderConfig(**values)


if __name__ == "__main__":
    unittest.main()
