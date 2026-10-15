**English** | [Русский](README.ru.md)

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
cargo run --release -- eval --gold eval/gold.en.jsonl --gold eval/gold.ru.jsonl \
    --store data/index_clean.sqlite=baseline --k 1,3,5,10

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

### Gold set

`eval/gold.en.jsonl` (71 questions) and `eval/gold.ru.jsonl` (25 questions) — one
JSON object per line:

```json
{
  "id": "en-lb-05",
  "question": "What does the variable LBORRES represent in the LB domain?",
  "lang": "en",
  "topic": "LB",
  "answerable": true,
  "expected": [{"source": "SDTMIG_v3.4.xlsx", "sheet": "Variables", "row": 892}],
  "reference_answer": "Result or Finding in Original Units (Char, role: Result Qualifier). ...",
  "needs_review": true,
  "notes": "auto: SDTMIG_v3.4.xlsx Variables row 892 (LB domain)"
}
```

A locator is a *place in the document* rather than a specific chunk: a file plus a
page, a section, or an Excel sheet+row. This makes the same set applicable to
indexes with different chunking, and the metrics independent of how the text was
split. Locator comparison is insensitive to case, whitespace and typography, so it
works on both the "raw" and the cleaned index.

Composition of the set:

| Topic | Questions | Gold source |
|---|---:|---|
| LB (laboratory test domain variables) | 36 | SDTMIG v3.4, Variables sheet |
| SDTM general model | 17 | SDTM v2.1, Variables sheet |
| PDF sections | 25 | SDTMIG v3.4 / SDTM v2.1 |
| Dataset structure | 11 | SDTMIG v3.4, Datasets sheet |
| Outside the documentation (`answerable: false`) | 7 | — |

Questions marked `needs_review: true` were generated automatically from the CDISC
definitions and require expert review; the PDF-section questions were written by
hand. Every locator is checked programmatically: it must intersect at least one
chunk of the index.

### Metrics

Retrieval metrics (computed offline, with no API calls):

- **Hit@k** — share of questions with at least one correct fragment in the top k;
- **Recall@k** — share of gold locators covered;
- **Precision@k** — share of relevant chunks in the top k;
- **nDCG@k** — binary relevance, accounts for the position of the correct fragment;
- **MRR** — reciprocal rank of the first correct fragment;
- response latency (mean / p50 / p95 / max).

Metrics are broken down by question language and by topic.

Answers (only with the `--judge` flag; requires `DEEPSEEK_API_KEY`) are scored from
0 to 1 by an LLM judge:

- **faithfulness** — share of the answer's claims supported by the supplied excerpts;
- **citation_accuracy** — whether the `[1]`, `[2]` markers match the excerpts they are attached to;
- **correctness** — agreement with the reference answer; for questions outside the
  documentation, an explicit refusal counts as correct;
- **refusal_correct** — whether the system refused to answer correctly.

### Ablations

Ready-to-use configurations for comparison live in `eval/configs/`:

| Config | Cleaning | Chunking | What it isolates |
|---|---|---|---|
| `raw.toml` | off | `target_tokens = 900` | baseline on raw text |
| `clean.toml` | on | same as `raw.toml` | the effect of text cleaning |
| `clean_tuned.toml` | on | `target_tokens = 400`, `min_chunk_chars = 600` | the effect of fitting chunks to the encoder window |

```bash
for c in raw clean clean_tuned; do
  sdtmrag index --rebuild --config eval/configs/$c.toml
done

sdtmrag eval --gold eval/gold.en.jsonl --gold eval/gold.ru.jsonl \
  --store data/index_raw.sqlite=raw \
  --store data/index_clean.sqlite=clean \
  --store data/index_clean_tuned.sqlite=clean+tuned
```

An index carries its own metadata (embedding model, vector dimension, chunking and
cleaning parameters, cleaning statistics, list of source files), so both evaluation
and search take the model from the index rather than from the current configuration
file, and `sdtmrag stats` shows the provenance of each index. If the query vector
dimension does not match the index vector dimension, the search fails with an
explicit error — otherwise comparing vectors of different lengths would silently
produce a wrong result.

Indexes used in the reports:

| File | What it is |
|---|---|
| `data/index_clean.sqlite` | the best measured configuration (cleaning on); used by default |
| `data/index_raw.sqlite` | cleaning off — the baseline |
| `data/index_clean_tuned.sqlite` | cleaning + chunks fitted to the encoder window |
| `data/index_clean_e5.sqlite` | cleaning + multilingual-e5-small (the cross-lingual slice) |
| `data/index_legacy.sqlite` | build predating the section-heading fix (6,296 chunks); kept as a reference point |

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
| `raw` (no cleaning) | 3,461 | 22.5 min | 10.6 MB | 80 ms |
| `clean` | 3,418 | 15.8 min | 10.3 MB | 68 ms |
| `clean+tuned` | 4,111 | 41.6 min | 12.2 MB | 65 ms |

Search without reranking takes tens of milliseconds. **Reranking is orders of
magnitude more expensive**: with 20 candidates and a query+passage length of 256
tokens it takes about 18 s per query; with 50 candidates and 512 tokens, about
130 s. The reason is that cross-encoder cost is linear in both the number of
candidates and the passage length, and bge-reranker-base is a 110M-parameter model
running on CPU. For interactive use, reranking has to be bounded
(`rerank.candidates`, `rerank.max_length`) or moved to a GPU.

## Evaluation results

Evaluation on the 89 answerable questions of the gold set, without reranking
(`eval/reports/`):

| Configuration | Hit@1 | Hit@5 | Recall@5 | nDCG@5 | MRR |
|---|---:|---:|---:|---:|---:|
| `raw` | 0.427 | 0.730 | 0.730 | 0.582 | 0.545 |
| `clean` | 0.461 | **0.775** | **0.775** | **0.624** | **0.584** |
| `clean+tuned` | **0.472** | 0.719 | 0.719 | 0.601 | 0.577 |

Text cleaning yields a statistically distinguishable gain (paired bootstrap:
Recall@5 +0.045 [0.011; 0.090], nDCG@5 +0.042 [0.006; 0.086]). Fitting the chunk
size to the encoder window produced no significant gain. The gap between question
languages is nearly twofold: Recall@5 = 0.879 for English questions versus 0.478
for Russian — a consequence of the monolingual embedding model.

Switching to a multilingual model (same corpus, cleaning and chunking) confirms
this and closes about half the gap (`eval/reports/multilingual/`):

| Embedding model | Recall@5 (en) | Recall@5 (ru) | MRR (ru) | MRR (all) |
|---|---:|---:|---:|---:|
| bge-small-en-v1.5 | 0.879 | 0.478 | 0.297 | 0.584 |
| multilingual-e5-small | **0.894** | **0.696** | **0.568** | **0.655** |

For Russian questions the gain is significant (Hit@1 +0.304 [0.130; 0.478],
McNemar p = 0.0156; nDCG@5 +0.260 [0.082; 0.441]), while English questions do not
degrade.

Effect of reranking (89 answerable questions, 20 candidates, 256 tokens): Recall@5
rises from 0.730 to 0.820 (`raw`), from 0.775 to 0.854 (`clean`) and from 0.719 to
0.798 (`clean+tuned`); nDCG@5 gains +0.066…+0.094; the first position improves too
(Hit@1 0.461 → 0.539 for `clean`). So the cross-encoder not only adds recall but
ranks what was found more accurately, at roughly a thousand times the cost of
vector search.

LLM-judge answer scoring (48 questions, `clean+tuned`): faithfulness 0.999,
citation accuracy 0.998, correctness 0.979, 4 of 4 correct refusals. The system
stays within the supplied excerpts and refuses to answer questions outside the
documentation.

Full reports (`report.json` and `report.md`) live in `eval/reports/`; confidence
intervals and paired tests are provided by `eval/analyze.py`:

```bash
python eval/analyze.py --pair clean:raw --pair clean+tuned:clean \
    eval/reports/rerank/<date>/report.json

# compare runs on a common subset
python eval/analyze.py --only-gold eval/gold.judge.jsonl \
    eval/reports/norerank/<date>/report.json eval/reports/judge/<date>/report.json
```

### Expert validation

The LLM-judge scoring and the gold set were produced by the authors of the system;
publishing the results requires independent review by domain specialists. The
worksheet for it is generated by a script, and the instructions for the reviewer
are in `eval/examples/review_instructions.md`:

```bash
python eval/make_review_sheet.py \
    eval/reports/judge/<date>/report.json \
    --gold eval/gold.en.jsonl --gold eval/gold.ru.jsonl \
    --out eval/review_sheet.csv --limit 25
```

Examples of real interactions with the system (a successful answer, a false refusal
caused by a cross-lingual retrieval failure, and a correct refusal) are analysed in
`eval/examples/clinical_examples.md`.

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
