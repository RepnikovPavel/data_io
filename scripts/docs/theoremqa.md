# theoremqa

**Script:** `pipe/clean_platypus/clean_theoremqa.py`

## Purpose

TheoremQA university-level math/science questions (test split). Rows containing a picture are dropped; the rest are direct question->answer pairs.

## Before (raw storage)

- Source: HF `TIGER-Lab/TheoremQA` (split `test`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `Question` | string | the question |
| `Answer` | string | the answer |
| `Picture` | string|null | filter only: rows with a picture are dropped |

## After (transformed)

- Location: `data/Platypus/theoremqa.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 747

`condition` values used here:

- `direct` — final answer only

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `TIGER-Lab/TheoremQA`, split `test`)

````text
{
  "Question": "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?",
  "Answer": "11760",
  "Picture": null
}
````

### Transformed record (`data/Platypus/theoremqa.jsonl`, record 1)

````text
{
  "condition": "direct",
  "instruction": "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?",
  "response": "11760"
}
````

### Transformed record (`data/Platypus/theoremqa.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "what is the value of $\\int_{-infty}^{+infty} sin(3*t)*sin(t/\\pi)/t^2 dt$?",
  "response": "1.0"
}
````
