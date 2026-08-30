# dmmath

↑ [index](README.md) · ← [synth](synth.md) · [next → acereason](acereason.md)

**Script:** `pipe_clustered/clean_dmmath.py`

## Purpose

DeepMind mathematics_dataset-v1.0: procedurally generated school math over ~120 task types, split into train-easy/medium/hard tiers for curriculum training. Each .txt holds alternating question/answer lines; one output parquet per (tier, task).

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/mathematics_dataset-v1.0/{train-easy,train-medium,train-hard}/*.txt`
- Storage: plain UTF-8 text: line 2k = question, line 2k+1 = answer

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question_line` | UTF-8 text line | odd lines: the question |
| `answer_line` | UTF-8 text line | even lines: the short answer |

## After (transformed)

- Location: `data_clustered/dmmath/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 168 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 111,999,888

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the target |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 7,999,992 | generated short answer | unlimited-volume clean drill data |

_exact within the first 12 of 168 files (7,999,992 of 111,999,888 rows); files are homogeneous, so the mix is representative_

## Keyword: `difficulty tier` (input)

Directory tier, encoded in the output filename prefix (train-easy__... etc.).

| value | rows | meaning | why it matters |
|---|---|---|---|
| `train-easy` | 37,333,296 | easy tier | curriculum stage 1 |
| `train-hard` | 37,333,296 | hard tier | curriculum stage 3 |
| `train-medium` | 37,333,296 | medium tier | curriculum stage 2 |

_exact (rows summed per file from metadata/line counts)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt`

First 4 lines of `/mnt/hdd2/datasets_text/mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt` (UTF-8 text; line 2k = question, line 2k+1 = answer):

````text
Solve 0 = 4*b + b + 15 for b.
-3
Solve -3*d = -0*d + 3 for d.
-1
````

### Transformed — condition=`direct`

One row of `data_clustered/dmmath/train-easy__algebra__linear_1d.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| Solve 0 = 4*b + b + 15 for b. | -3 | direct |

---

↑ [index](README.md) · ← [synth](synth.md) · [next → acereason](acereason.md)
