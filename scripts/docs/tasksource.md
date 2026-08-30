# tasksource

↑ [index](README.md) · ← [sudoku_extreme](sudoku_extreme.md) · [next → textbookreasoning](textbookreasoning.md)

**Script:** `pipe_clustered/clean_tasksource.py`

## Purpose

tasksource-instruct-v0: 5.3M instruction examples recast from 485 curated HF datasets (mostly discriminative: NLI, classification, tagging, MCQ), capped at 30k rows per task. The transform keeps a curated ~180-task subset and writes one parquet per task.

## Before (raw storage)

- Source: HF `tasksource/tasksource-instruct-v0` (split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `task` | UTF-8 text (arrow `string`) | task id; filter (TASK_SET) and output filename |
| `inputs` | UTF-8 text (arrow `string`) | prompt -> instruction |
| `targets` | UTF-8 text (arrow `string`) | target -> response (trailing '.' removed) |

## After (transformed)

- Location: `data_clustered/tasksource/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 182 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 2,363,550

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the target |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 131,201 | short task target | broad discriminative-task supervision |

_exact within the first 12 of 182 files (131,201 of 2,363,550 rows); files are homogeneous, so the mix is representative_

## Keyword: `task` (output)

One output parquet per kept task; the task name is the filename. High cardinality — counts as file count only.

| value | rows |
|---|---|
| `distinct tasks (one parquet file each)` | 182 |

_exact (output file count)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `tasksource/tasksource-instruct-v0`, split `train`

One row of HF `tasksource/tasksource-instruct-v0`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| task | inputs | targets |
|---|---|---|
| glue/mnli | With no explanation, label text_A→text_B with either "entailment", "neutral" or "contradiction".⏎text_A: After learning that one of its members had been taken in by the scheme, the Middle East Studies Association posted a warning on its Web site.⏎text_B: A member of the Middle East Studies Association was scammed for money. | neutral. |

### Transformed — condition=`direct`

One row of `data_clustered/tasksource/CONDAQA.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| With no explanation, label text_A→text_B with either "DON'T KNOW", "NO" or "YES".⏎text_A: In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While he vetoed what was an unprecedented eleven City Council ordinances that year, most narrowly were involved with specific financial practices considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimou… [truncated, 798 chars total] | NO | direct |

---

↑ [index](README.md) · ← [sudoku_extreme](sudoku_extreme.md) · [next → textbookreasoning](textbookreasoning.md)
