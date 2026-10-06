# External dataset and acquisition

The selected external dataset is **DReggio/esg-spanish-events**, published by
DReggio: https://huggingface.co/datasets/DReggio/esg-spanish-events

The repository indicates **CC BY-NC 4.0**. Preserve dataset attribution, source,
and the license link: https://creativecommons.org/licenses/by-nc/4.0/
The dataset's noncommercial condition applies to its use. This project's code
and the external dataset are separate works. We did not create or acquire
ownership of the dataset and do not assign a license to third-party data.
No project-code license has been selected.

## Reproducible acquisition

Run from the project root with standard Python; no account, token, extra package,
or archival directory is required:

```powershell
$env:PYTHONPATH = 'src'
python -B -m semantic_search_spanish.acquisition
```

Only three root-level gold CSVs are requested from repository commit
`e900148694757d37df41035df090dd640052874d`. No silver data is requested.
Raw downloads go to ignored `data/raw/esg-spanish-events/`.

| File | Expected SHA-256 |
| --- | --- |
| gold_train.csv | ef529adcae0a1c2138d249d35a61e58b57f8ecdc793778567f31947fed43a256 |
| gold_dev.csv | 1ccc62efe4cee7a2ff1f13a49cb4868f8473050960a8c03ca99f859ae901d0b5 |
| gold_test.csv | cdbfeb7c22b1913d5b0047e17bfd6b7ce54d1b484bda6e47cb690f7a932096ee |

Hashes were checked against anonymous remote responses at the pinned commit
and match the local archival files. Each download is verified before its final
rename. Existing valid files are reused; existing mismatches cause an error
without replacement. Partial failures may leave earlier verified files.
The downloader rejects destinations inside `_originals/`, including resolved
paths. Custom destinations should remain in ignored local storage.

To report statistics without network access:

```powershell
python -B -m semantic_search_spanish.acquisition --stats-only
```

## Preparation and provenance

`load_corpus(directory)` reads all three splits in train/dev/test order. It uses
`canonical_headline` and preserves the source `event_id`, split, one-based data
row, and original text. It does not interpret ESG, sentiment, company, sector,
or other metadata as relevance. Source CSVs are never edited or manually merged.

Cleaning applies Unicode NFC and collapses whitespace, preserving case,
accents, and punctuation. Missing/blank headlines are counted and omitted.
Normalized exact duplicate headlines form one retrieval document. The first
record by split precedence and row order supplies its stable event_id; all
contributing records remain in `PreparedCorpus.provenance`. Distinct event IDs
are not discarded from provenance. Duplicate event IDs with conflicting text
are rejected. No fuzzy or semantic deduplication is performed.

Combining these original classification splits defines a retrieval corpus.
It does not supply a retrieval training/test protocol or relevance judgments.
The official evaluation has 15 frozen queries and 43 human positive judgments
over 648 blinded pooled query-document pairs; see ../evaluation/README.md.

## Verified schema and descriptive counts

All three source files have these same 21 ordered columns:

```text
event_id, canonical_headline, company_name, company_nif, company_ticker,
sics_sector, sics_industry, event_date, has_cnmv, source_name, pillar_profile,
esg_E, esg_S, esg_G, esg_none, sentiment, annotator_confidence, comment,
annotation_date_nessie, annotation_date, review_date
```

CSV values are read as strings. Only the identifier and headline are used in
preparation. Metadata types are not inferred or coerced.

| Scope | Rows | Missing headlines | Exact duplicate groups / extra rows | Normalized duplicate groups / extra rows | Duplicate ID extra rows | Unique normalized headlines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1243 | 0 | 25 / 39 | 25 / 39 | 0 | 1204 |
| dev | 178 | 0 | 3 / 3 | 3 / 3 | 0 | 175 |
| test | 267 | 0 | 0 / 0 | 0 / 0 | 0 | 267 |
| combined | 1688 | 0 | 39 / 63 | 39 / 63 | 0 | 1625 |

Combined duplicate headline groups contain 102 source rows. Both exact and
normalized cross-split duplication comprise 18 groups containing 53 source
rows. These cross-split groups overlap within-split duplicates and must not be
added to them. The prepared in-memory corpus contains **1625 documents**.
These are corpus statistics, not retrieval benchmark results.
