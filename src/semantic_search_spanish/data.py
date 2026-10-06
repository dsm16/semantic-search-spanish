"""Validation and conservative cleaning; source files are never written."""

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    original_text: str


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving case and accents."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    text = " ".join(unicodedata.normalize("NFC", text).split())
    if not text:
        raise ValueError("Text must not be blank")
    return text


def clean_documents(texts: Iterable[str], *, ids: Iterable[str] | None = None,
                    deduplicate: bool = True) -> list[Document]:
    """Reject invalid text; retain first normalized exact duplicate.

    IDs default to zero-based input positions and survive deduplication.
    Matching is case-sensitive; no stemming or accent removal is performed.
    """
    if isinstance(texts, (str, bytes)):
        raise ValueError("Expected a collection of texts")
    texts = list(texts)
    ids = list(ids) if ids is not None else [str(i) for i in range(len(texts))]
    if len(ids) != len(texts):
        raise ValueError("IDs and texts must have equal lengths")
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("IDs must be nonblank strings")
    if len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique")
    documents, seen = [], set()
    for identifier, original in zip(ids, texts):
        text = normalize_text(original)
        if not deduplicate or text not in seen:
            documents.append(Document(identifier, text, original))
        seen.add(text)
    return documents


def load_csv(path: str | Path, *, text_column: str = "title",
             id_column: str | None = None, deduplicate: bool = True) -> list[Document]:
    """Read a user-supplied UTF-8 CSV; reject malformed rows and missing columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {text_column} | ({id_column} if id_column is not None else set())
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV is missing required columns")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV has duplicate column names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV has malformed rows")
    return clean_documents(
        [row[text_column] for row in rows],
        ids=[row[id_column] for row in rows] if id_column is not None else None,
        deduplicate=deduplicate,
    )
