"""Exactly six experiment configurations; construction is explicit and lazy."""

from .dense import AutoModelEncoder, DenseRetriever, EncoderConfig, SentenceEncoder
from .encoders import TfidfRetriever

BETO = "dccuchile/bert-base-spanish-wwm-cased"
ROBERTUITO = "pysentimiento/robertuito-base-cased"
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SYSTEMS = {
    "tfidf": None,
    "beto_first_token": EncoderConfig(BETO, "first_token"),
    "beto_masked_mean": EncoderConfig(BETO, "masked_mean"),
    "robertuito_first_token": EncoderConfig(ROBERTUITO, "first_token"),
    "robertuito_masked_mean": EncoderConfig(ROBERTUITO, "masked_mean"),
    "minilm": EncoderConfig(MINILM, "sentence_transformer"),
}


def build_system(name, documents, *, local_files_only=True):
    """Use cached models by default; never download unless explicitly opted in."""
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system: {name}")
    config = SYSTEMS[name]
    if config is None:
        return TfidfRetriever(documents)
    encoder_class = SentenceEncoder if config.pooling == "sentence_transformer" else AutoModelEncoder
    return DenseRetriever(documents, encoder_class(config, local_files_only=local_files_only))
