"""Gold-only ESG corpus preparation. No metadata is used as relevance."""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data import Document, normalize_text

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class SourceHeadline:
    event_id: str
    split: str
    row: int  # one-based data row, excluding header
    original_text: str


@dataclass(frozen=True)
class PreparedCorpus:
    documents: tuple[Document, ...]
    provenance: dict[str, tuple[SourceHeadline, ...]]
    statistics: dict


def load_gold(directory: str | Path) -> tuple[SourceHeadline, ...]:
    """Read exactly gold_train/dev/test.csv, preserving source row order.

    Missing headlines remain available for reporting; missing IDs and malformed
    CSV structure raise errors. Additional metadata is not loaded into records.
    """
    records = []
    schema = None
    for split in SPLITS:
        with (Path(directory) / f"gold_{split}.csv").open(
                encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if (fields is None or len(set(fields)) != len(fields)
                    or not {"event_id", "canonical_headline"}.issubset(fields)):
                raise ValueError(f"Invalid gold {split} schema")
            if schema is not None and fields != schema:
                raise ValueError("Gold split schemas must match")
            schema = fields
            for row_number, row in enumerate(reader, 1):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"Malformed gold {split} row {row_number}")
                identifier = row["event_id"]
                if not identifier.strip() or identifier != identifier.strip():
                    raise ValueError(f"Invalid event ID in {split} row {row_number}")
                records.append(SourceHeadline(identifier, split, row_number,
                                              row["canonical_headline"]))
    return tuple(records)


def _duplicate_counts(values) -> dict:
    counts = Counter(values)
    return {"groups": sum(n > 1 for n in counts.values()),
            "extra_rows": sum(n - 1 for n in counts.values()),
            "rows_in_groups": sum(n for n in counts.values() if n > 1)}


def corpus_statistics(records) -> dict:
    """Counts only; raw headlines are never embedded in the statistics report.

    Duplicate counts exclude missing headlines. Normalization is NFC plus
    collapsed whitespace, case-sensitive and accent-preserving.
    """
    records = tuple(records)
    def summarize(items):
        valid = [r for r in items if r.original_text.strip()]
        return {"rows": len(items), "missing_headlines": len(items) - len(valid),
                "exact_duplicates": _duplicate_counts(r.original_text for r in valid),
                "normalized_duplicates": _duplicate_counts(
                    normalize_text(r.original_text) for r in valid),
                "duplicate_event_ids": _duplicate_counts(r.event_id for r in items),
                "unique_normalized_headlines": len({normalize_text(r.original_text)
                                                    for r in valid})}
    def cross_split(normalized):
        groups = defaultdict(list)
        for r in records:
            if r.original_text.strip():
                key = normalize_text(r.original_text) if normalized else r.original_text
                groups[key].append(r)
        shared = [g for g in groups.values() if len({r.split for r in g}) > 1]
        return {"groups": len(shared), "rows_in_groups": sum(map(len, shared))}
    return {"combined": summarize(records),
            "splits": {s: summarize([r for r in records if r.split == s]) for s in SPLITS},
            "cross_split_exact_headlines": cross_split(False),
            "cross_split_normalized_headlines": cross_split(True)}


def prepare_corpus(records) -> PreparedCorpus:
    """Deduplicate normalized headlines; retain every source reference.

    Stable precedence is train, dev, test, then source row. The representative
    document ID is the first source event_id, never a generated positional ID.
    Missing headlines are reported and omitted. An event_id associated with
    conflicting normalized texts is rejected, including blank/nonblank conflicts.
    Combining original splits defines a retrieval corpus, not an evaluation split.
    """
    records = tuple(records)
    if any(r.split not in SPLITS or not r.event_id.strip() or r.row <= 0
           for r in records):
        raise ValueError("Invalid source provenance")
    ordered = sorted(records, key=lambda r: (SPLITS.index(r.split), r.row, r.event_id))
    by_id, groups = {}, {}
    for record in ordered:
        text = normalize_text(record.original_text) if record.original_text.strip() else ""
        if record.event_id in by_id and by_id[record.event_id] != text:
            raise ValueError(f"Conflicting headlines for event_id {record.event_id}")
        by_id[record.event_id] = text
        if text:
            groups.setdefault(text, []).append(record)
    documents = tuple(Document(group[0].event_id, text, group[0].original_text)
                      for text, group in groups.items())
    provenance = {group[0].event_id: tuple(group) for group in groups.values()}
    statistics = corpus_statistics(records)
    statistics["prepared_documents"] = len(documents)
    return PreparedCorpus(documents, provenance, statistics)


def load_corpus(directory: str | Path) -> PreparedCorpus:
    return prepare_corpus(load_gold(directory))
