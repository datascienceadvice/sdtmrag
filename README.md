# sdtmrag

RAG over the CDISC SDTM / SDTMIG documentation. Local embeddings and reranking (ONNX), answer generation via the DeepSeek API, CLI + REST API.

The project answers questions about the SDTM standard and the SDTM Implementation
Guide (SDTMIG) — including the laboratory test results (LB) domain — citing the
source documents. Particular attention is paid to the two things almost everything
in such systems depends on: **the quality of text extracted from PDF**, and
**the measurability of retrieval and answer quality**.

## Requirements

- Rust 1.85+ (the crate uses edition 2024)
- Windows / Linux (tested on Windows)
- ~2 GB of free disk space for the ONNX models (downloaded automatically on first use)

## Source documents

The repository **does not contain** the CDISC documents: they are distributed under
the CDISC licence and cannot be included in a public repository. Before the first
run, download them from
[cdisc.org](https://www.cdisc.org/standards/foundational/sdtm) and put them in
`dicts/`:

| File | Where to get it |
|---|---|
| SDTM v2.1 (Final) — PDF | SDTM section → version 2.1 |
| SDTM_v2.1.xlsx | same place, tabular variable definitions |
| SDTMIG v3.4 (Final) — PDF | SDTMIG section → version 3.4 |
| SDTMIG_v3.4.xlsx | same place, domain variable definitions |

The `dicts/` directory is excluded from Git (`.gitignore`), so your copies of the
documents will not end up in the repository. Any other `.pdf`/`.xlsx` files are
supported as well — the pipeline is not tied to these four files.

## Quick start

```bash
# build
cargo build --release

# put the documents into dicts/ (see the "Source documents" section)

# see what PDF text cleaning does (no index is built)
cargo run --release -- clean

# build the index (~10-15 min at ~8 chunks/s on a slow PC)
cargo run --release -- index --rebuild

# ask a question
cargo run --release -- query "What is SUPPQUAL used for?" --no-generate --top-k 3

# evaluate against the gold set
cargo run --release -- eval --gold eval/gold.en.jsonl \
    --store data/index_clean.sqlite=clean --k 1,3,5,10

# or build with the REST API
cargo build --release --features server
cargo run --release --features server -- serve
```

## CLI

Global options: `--config <path>` (`config.toml` if present, by default) and
`--store <path>` (overrides the index path).

### `index`

Build or rebuild the index from the files in `inputs` (`dicts/` by default).

```bash
sdtmrag index                    # incremental (a full build if no index exists)
sdtmrag index --rebuild          # drop the old index and build from scratch
sdtmrag index --inputs path/to   # specify the directory
```

Embedding progress is written to stderr every 128 chunks. Text-cleaning statistics
are printed at the end.

### `clean`

Shows the effect of PDF text cleaning without building an index. Useful when tuning
the cleaning rules for a new document.

```bash
sdtmrag clean                          # summary per PDF
sdtmrag clean --json                   # the same summary as JSON
sdtmrag clean --out data/cleaned       # dump the cleaned text, page by page
sdtmrag clean --limit 1                # first file only
```

### `query`

Search by question: retrieval -> rerank -> generation.

```bash
sdtmrag query "How to create a SUPPQUAL dataset?"
sdtmrag query "Define LBORRES" --top-k 3 --no-generate   # no generation, retrieval only
sdtmrag query "STUDYID" --no-rerank --json               # cosine search without reranking
```

### `eval`

Quality evaluation against the gold set of questions (see the "Evaluation"
section).

```bash
# compare several indexes in a single report
sdtmrag eval --gold eval/gold.en.jsonl \
    --store data/index_raw.sqlite=raw \
    --store data/index_clean.sqlite=clean \
    --store data/index_clean_tuned.sqlite=clean+tuned \
    --k 1,3,5,10 --out eval/reports

# add LLM-judge answer scoring (requires DEEPSEEK_API_KEY)
sdtmrag eval --gold eval/gold.en.jsonl --store data/index_clean.sqlite=clean --judge
```

The report is written to `eval/reports/<date>/{report.json,report.md}` and echoed to
stdout.

### `stats` / `list-docs`

```bash
sdtmrag stats       # chunk count, build metadata, list of sources
sdtmrag list-docs   # file + chunk count per source
```

### `serve`

REST API (axum). Requires the `server` feature.

```bash
sdtmrag serve                        # 127.0.0.1:8080
sdtmrag serve --bind 0.0.0.0 --port 9090
```

## REST API

| Method | Path            | Description                       |
|--------|-----------------|-----------------------------------|
| GET    | `/api/health`   | `{"status":"ok","chunks":6296}`   |
| GET    | `/api/stats`    | sources, models, settings         |
| GET    | `/api/documents`| documents present in the index    |
| POST   | `/api/query`    | retrieval + optional generation   |

### POST /api/query

```json
{
  "question": "What is SUPPQUAL?",
  "top_k": 6,
  "rerank": true,
  "generate": false
}
```

Response:
```json
{
  "answer": null,
  "model": "deepseek-chat",
  "sources": [
    {"score": 1.0, "source": "SDTM_v2.1.xlsx", "kind": "xlsx", "sheet": "Variables", "row": 508, "preview": "..."}
  ]
}
```

## PDF text cleaning

`pdf-extract` returns a "raw" stream that is unfit for embedding. Measured on a
corpus of two PDFs (545 pages):

| Problem | How it shows up |
|---|---|
| Footers inside lines | `... CRF  nsortium, Inc. All rights reserved  Page 258 2021-11-29 The SUPPBE dataset ...` — the copyright notice, page number and release date are glued to meaningful text |
| Collapsed tables | columns separated by runs of spaces: `2  ABC  BE  ABC-01-101  BEREFID ...` |
| Table of contents in the stream | lines such as `1.1  PURPOSE .............. 5` enter the text as ordinary lines |
| Broken hyphenation | `impl-` + `ementation` |
| Typography | ligatures, en dashes `–`, curly quotes `“ ”`, non-breaking spaces, soft hyphens |

The `src/clean.rs` module removes all of the above:

1. **Unicode normalisation** — ligatures (`ﬁ` → `fi`), dashes and quotes → ASCII,
   removal of `©`/`®`/`™`, nbsp, soft hyphens and replacement characters.
2. **Inline scrubbing** — regular expressions remove footers *glued inside*
   meaningful lines (the main source of noise: ~68 thousand characters on the corpus).
3. **Footer-line filtering** — copyright notices, bare page numbers, ISO release
   dates, running headers of the standard.
4. **Repeated-line removal** — a line occurring on at least
   `boilerplate_min_pages` pages, or on `boilerplate_page_ratio` of the pages, is
   kept only at its first occurrence (this removes repeating table headers).
5. **Table-of-contents stripping** — lines with dot leaders and CONTENTS headings.
6. **De-hyphenation** — joining words broken by a line-end hyphen. The decision uses
   the corpus vocabulary: if the hyphenated form occurs in the text as an ordinary
   word (`non-standard`) while the joined form does not, the hyphen is kept;
   otherwise the word is joined.
7. **Table-row markup** — lines with two or more runs of 2+ spaces are re-marked as
   `a | b | c`, which stops table rows from pretending to be section headings and
   preserves cell boundaries for the embedding.

Measured effect on the current corpus: 46,019 → 31,260 lines (−32%),
2,007,379 → 1,813,627 characters (−9.7%), of which 67,964 characters are footers
inside lines; 8,516 table rows re-marked, 817 footer lines, 1,612 repeated lines,
329 table-of-contents lines, 212 hyphen joins.

The chunker additionally checks that a section heading really is a heading: it
rejects lines containing `|`, lines with dot leaders, and lines with two or more
runs of 2+ spaces. Before this check, 82.7% of PDF chunks received an unreliable
section value (`1 | Role | CDISC Notes | Core`,
`0 mg  5  mm  5  5  mm  FOREARM 1  WEEK`), and the number of unique section values
reached 3,000 versus 372 after the fix; this is verified by
`eval/section_quality.py`.

## Evaluation

### Benchmark

`eval/gold.en.jsonl` holds 193 English questions, one JSON object per line:

```json
{
  "id": "en-lookup-06",
  "question": "Which LB variable records the unit in which the original result was reported, for example g/L?",
  "register": "lookup",
  "stratum": "tabular",
  "answerable": true,
  "expected": [
    {"source": "SDTMIG_v3.4.xlsx", "sheet": "Variables", "row": 893},
    {"source": "SDTMIG v3.4-FINAL_2022-07-21.pdf",
     "contains": "Original units in which the data were collected. The unit for LBORRES."}
  ],
  "any_of": true,
  "reference_answer": "Version: SDTMIG v3.4 ... CDISC Notes: Original units in which the data were collected. ...",
  "provenance": "index-extracted",
  "needs_review": true
}
```

Three properties make the benchmark independent of the pipeline it evaluates:

1. **A locator is content, not layout.** It is a source plus a verbatim passage, never
   a chunk id and never a section title. Section titles are themselves changed by the
   cleaning step, so a title-based locator would measure the preprocessing rather than
   the retrieval. Every locator is validated against *all* compared indexes before it
   is accepted (`eval/verify_gold.py`).
2. **A reference answer is extracted, not written.** It is the text at the locator, so
   its correctness does not depend on an annotator. The verification script checks that
   it occurs verbatim in the source.
3. **Duplicates are explicit.** Many variable definitions appear both as a workbook row
   and inside the specification table of the PDF. Such items carry an alternative
   locator and `any_of: true`: the answer counts if either version is retrieved.
   Without this, a configuration that retrieves the prose version would be penalised
   for finding the right answer.

Composition:

| Register | Questions | Locator | Target |
|---|---:|---|---|
| `lookup` | 55 | 1 (+1 alias) | a variable definition |
| `structure` | 20 | 1 (+1 alias) | a dataset description |
| `concept` | 69 | 1 | a narrative section |
| `procedural` | 29 | 2-3 | a multi-step task |
| `out-of-scope` | 20 | none | refusal |

Questions were written by the authors; answers and locators are produced by
`eval/build_gold.py` from the index. Since no laboratory specialist annotated the set,
its quality is assessed mechanically: `eval/check_gold.py` asks an independent model
whether each reference passage suffices to answer its question. 168 of 173 passages
(97.1%) passed; the five failures are listed in the report and are documented as a
limitation. The same check confirmed that all 20 unanswerable questions cannot be
answered from the CDISC documents.

### Metrics

Retrieval metrics are computed offline, with no API calls:

- **Hit@5** — share of questions with a correct passage in the top 5;
- **Recall@5** — share of gold locators covered (1 for `any_of` items if any alias is found);
- **Precision@k**, **nDCG@k** (`[8]` in the paper), **MRR** — position-aware metrics;
- response latency (mean / p50 / p95 / max).

Answer quality (only with `--judge`; requires `DEEPSEEK_API_KEY`) is scored 0-1 by an
LLM judge: **faithfulness**, **citation accuracy**, **correctness**, and for
unanswerable questions **refusal correctness**.

Inference-free baselines live next to the dense one:

| Baseline | Command |
|---|---|
| lexical (BM25 over the same chunks) | `--retriever bm25 --no-rerank` |
| dense (bge-small-en-v1.5) | default |
| fusion of both (RRF, k = 60) | `--retriever hybrid --no-rerank` |
| closed book (no retrieval at all) | `python eval/closedbook.py` |

### Ablations

| Config | Cleaning | Chunking | What it isolates |
|---|---|---|---|
| `raw.toml` | off | `target_tokens = 900` | baseline on raw text |
| `clean.toml` | on | same as `raw.toml` | the effect of text cleaning |
| `clean_tuned.toml` | on | `target_tokens = 400`, `min_chunk_chars = 600` | fitting chunks to the encoder window |

```bash
for c in raw clean clean_tuned; do
  sdtmrag index --rebuild --config eval/configs/$c.toml
done

python eval/build_gold.py          # extract answers, validate locators
python eval/verify_gold.py         # locators in every index, verbatim references
python eval/check_gold.py          # does each passage answer its question?

for r in dense bm25 hybrid; do
  sdtmrag eval --gold eval/gold.en.jsonl --retriever $r --no-rerank     --store data/index_raw.sqlite=raw     --store data/index_clean.sqlite=clean     --store data/index_clean_tuned.sqlite=clean+tuned     --out eval/reports/${r}_norerank
done
```

An index carries its own metadata (embedding model, vector dimension, chunking and
cleaning parameters, cleaning statistics, list of source files), so both evaluation and
search take the model from the index rather than from the current configuration file. If
the query vector dimension does not match the index, search fails with an explicit error
instead of silently returning a wrong ranking.

| File | What it is |
|---|---|
| `data/index_clean.sqlite` | cleaning on; the primary index |
| `data/index_raw.sqlite` | cleaning off — the baseline |
| `data/index_clean_tuned.sqlite` | cleaning + chunks fitted to the encoder window |

Earlier material of the previous, bilingual version of this study (a Russian gold set, a
multilingual embedding configuration, the reports produced with it and the Russian README)
is kept in `eval/legacy/` for reference and is not used in the current evaluation. Those
numbers were produced with the earlier benchmark and are superseded.

## Configuration

Copy `config.toml.example` to `config.toml`. All parameters are optional (defaults
exist). The sections are documented inside the file.

## Architecture

```
dicts/*.pdf,*.xlsx
       |
    extract       pdf-extract (page by page) / calamine (row by row)
       |
     clean        footer, TOC and duplicate removal; de-hyphenation;
                  Unicode normalisation; table-row markup      (src/clean.rs)
       |
     chunk        PDF: section regex + heading validation, merging of short
                  chunks; XLSX: row by row
       |
     embed        fastembed (ONNX) -> normalised f32 vectors, dim=384
       |
     store        SQLite: chunks(source, kind, page, section, sheet, row, text,
                  embedding BLOB) + meta (index provenance)
       |
    query        cosine brute-force -> top candidates -> cross-encoder rerank -> context
       |
  generate       DeepSeek API (chat completions)
       |
     eval        locators -> Hit@k / Recall@k / nDCG@k / MRR (+ LLM judge)
```

## Models (auto-download)

| Component   | Default model       | Size   | Alternatives                             |
|-------------|---------------------|--------|------------------------------------------|
| Embedding   | bge-small-en-v1.5   | ~130 MB| bge-base-en-v1.5, bge-large-en-v1.5, bge-m3, multilingual-e5-small/base/large, paraphrase-multilingual-minilm-l12-v2, all-minilm-l6-v2 |
| Reranking   | bge-reranker-base   | ~1.1 GB| bge-reranker-v2-m3, jina-reranker-v1-turbo-en |

Models are downloaded into `models/` on first use.

Instruction prefixes are applied automatically: for the bge family the retrieval
instruction is prepended to the query; for the E5 family, `query: ` is prepended to
the query and `passage: ` to documents (without them E5 retrieval quality drops
noticeably).

**Model limitations worth keeping in mind.**

* bge-small-en-v1.5 is trained on English and reads at most 512 tokens
  (~2,000 characters). A chunk longer than the window is silently truncated at
  embedding time — indexing prints a warning when `target_tokens * 4` exceeds the
  window. Russian-language questions require a multilingual model.
* In fastembed, bge-m3 ships as fp32 ONNX (~2.2 GB, 568M parameters) and delivers
  about 0.04 chunks/s on CPU, i.e. ~23 hours for a corpus of 3,418 chunks. For
  multilingual retrieval on CPU, `multilingual-e5-small` (118M parameters,
  384 dimensions) or `paraphrase-multilingual-minilm-l12-v2` is more practical.
* Thread count and batch size are set in `[embedding]` (`intra_threads`,
  `batch_size`); for large models the batch should be reduced.

## Performance

Measured on a desktop CPU (8 logical cores, `intra_threads = 4` for embedding),
without isolation from other processes:

| Configuration | Chunks | Indexing | Index size | Search (p50) |
|---|---:|---:|---:|---:|
| `raw` (no cleaning) | 3,437 | 18.2 min | 10.1 MB | 24 ms |
| `clean` | 3,395 | 17.1 min | 9.8 MB | 23 ms |
| `clean+tuned` | 4,092 | 22.9 min | 11.6 MB | 26 ms |

Search without reranking takes tens of milliseconds. **Reranking is orders of
magnitude more expensive**: with 20 candidates and a query+passage length of 256
tokens it takes about 13 s per query on the corpus used here; with 50 candidates and
512 tokens, about 130 s. The reason is that cross-encoder cost is linear in both the number of
candidates and the passage length, and bge-reranker-base is a 110M-parameter model
running on CPU. For interactive use, reranking has to be bounded
(`rerank.candidates`, `rerank.max_length`) or moved to a GPU.

## Evaluation results

Primary runs (`eval/reports/`), 173 answerable questions, no reranking:

| Retriever | Index | Hit@5 | Recall@5 | nDCG@5 | MRR |
|---|---|---:|---:|---:|---:|
| BM25 | raw | 0.711 | 0.688 | 0.618 | 0.504 |
| BM25 | clean | 0.699 | 0.679 | 0.610 | 0.510 |
| BM25 | clean+tuned | 0.653 | 0.636 | 0.591 | 0.481 |
| dense | raw | **0.723** | **0.686** | 0.606 | 0.522 |
| dense | clean | 0.682 | 0.643 | 0.584 | 0.532 |
| dense | clean+tuned | 0.705 | 0.676 | 0.610 | 0.556 |
| hybrid | raw | 0.792 | 0.750 | 0.672 | 0.581 |
| hybrid | clean | **0.803** | **0.759** | **0.685** | 0.589 |
| hybrid | clean+tuned | 0.786 | 0.747 | 0.687 | **0.603** |

What the numbers say:

- **Fusion is the only large effect.** Hybrid retrieval beats dense retrieval by
  +0.121 Hit@5 [0.058; 0.185], McNemar P = 0.0005 (Holm-adjusted 0.002), and this
  survives correction for the four planned primary comparisons.
- **Text cleaning does not improve retrieval.** Hit@5 is 0.682 with cleaning against
  0.723 without (difference -0.040 [-0.075; -0.012], unadjusted P = 0.039,
  Holm-adjusted P = 0.08).
- **Citation metadata was fixed by the chunker, not by cleaning.** An earlier build gave
  82.7% of PDF chunks an unreliable section label (3,000 distinct labels for 3,707
  chunks), because table rows beginning with a section number were accepted as chapter
  headings. After that rule was added the share is 5.6% without cleaning and 4.7% with
  it, so cleaning accounts for less than one of the 77 percentage points.
- **Dense is not better than lexical.** Hit@5 0.682 versus 0.699 (P = 0.79), MRR 0.532
  versus 0.510: on a corpus of short technical definitions, term matching carries most
  of the signal.
- **Chunk-size tuning does nothing measurable**, although 900-token chunks exceed the
  512-token encoder window and are truncated at embedding time.

Reranking (`eval/reports/dense_rerank/`, cleaned index): Hit@5 rises from 0.682 to
0.734, nDCG@5 from 0.584 to 0.622 and MRR from 0.532 to 0.576, but not significantly
(difference 0.052 [-0.006; 0.110], P = 0.12) and at a median latency of 12.9 s against
23 ms. Fusion without any reranking reaches Hit@5 0.803 at 23 ms, so for interactive use
it is the better trade.

Answer quality and refusal (`eval/reports/judge_rerank/`, 108-question stratified
subsample): faithfulness 0.99, citation accuracy 0.98, correctness 0.89, and correct refusal on all 19 unanswerable questions that were judged (two of the 108 items failed at the API and are excluded).

Closed book (`eval/reports/closedbook/`, all 193 questions): correctness 0.61 on
answerable questions, and only 65% of unanswerable questions refused — the same model
answers 35% of questions the documentation does not cover when it has no passages to
ground on.

Tables and figures for the paper are generated from the reports:

```bash
python eval/tables.py --dir eval/reports/dense_norerank     --dir eval/reports/bm25_norerank --dir eval/reports/hybrid_norerank     --out article/tables.md --json article/numbers.json
python eval/figures.py
```

## Citation

If the software is used in academic work, please cite it via `CITATION.cff` (GitHub
shows a "Cite this repository" button). The run parameters and the resulting numbers
are described in the paper; the full reports are in `eval/reports/`.

## Licence and legal information

The code is released under the MIT licence (see `LICENSE`).

The CDISC documents (SDTM, SDTMIG) are **not included** in the repository and are
distributed under a separate CDISC licence; download them yourself before use (see
the "Source documents" section). CDISC, SDTM and SDTMIG are trademarks of Clinical
Data Interchange Standards Consortium, Inc. This project is neither approved nor
certified by CDISC.

**This software is not a medical device** and is not intended for clinical
diagnosis, treatment decisions or any other medical use. Answers are generated by a
language model, may contain errors, and always require verification against the
primary source — the section and page references are provided for exactly that
purpose.
