# dmmath

**Script:** `pipe_clustered/clean_dmmath.py`

## Purpose

DeepMind mathematics_dataset-v1.0: procedurally generated school-level math across ~120 task types and 3 difficulty tiers. Each .txt file holds alternating question/answer lines; one output parquet per (tier, task).

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/mathematics_dataset-v1.0/{train-easy,train-medium,train-hard}/*.txt`
- Format: plain text: line 2k = question, line 2k+1 = answer

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question_line` | string | odd lines: the question |
| `answer_line` | string | even lines: the short answer |

## After (transformed)

- Location: `data_clustered/dmmath/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 168 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 111,999,888

`condition` values used here:

- `direct` — generated short answer

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/mathematics_dataset-v1.0/train-easy/algebra__linear_1d.txt`)

````text
{
  "question_line": "Solve 0 = 4*b + b + 15 for b.",
  "answer_line": "-3"
}
````

### Transformed record (`data_clustered/dmmath/train-easy__algebra__linear_1d.parquet`, record 1)

````text
{
  "instruction": "Solve 0 = 4*b + b + 15 for b.",
  "response": "-3",
  "condition": "direct"
}
````

### Transformed record (`data_clustered/dmmath/train-easy__algebra__linear_1d.parquet`, record 2)

````text
{
  "instruction": "Solve -3*d = -0*d + 3 for d.",
  "response": "-1",
  "condition": "direct"
}
````
