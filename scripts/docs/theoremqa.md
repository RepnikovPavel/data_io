# theoremqa

↑ [index](README.md) · ← [scienceqa](scienceqa.md) · [next → flan](flan.md)

**Script:** `pipe/clean_platypus/clean_theoremqa.py`

## Purpose

TheoremQA: 800 expert-curated university-level questions driven by 350+ STEM theorems (math, EE&CS, physics, finance). Rows containing a picture are dropped; the rest become direct question->answer pairs.

## Before (raw storage)

- Source: HF `TIGER-Lab/TheoremQA` (split `test`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `Question` | UTF-8 text (arrow `string`) | the question |
| `Answer` | UTF-8 text (arrow `string`) | the answer |
| `Picture` | image (arrow `struct<bytes, path>`, PIL-decoded) | filter only: rows with a picture are dropped |

## After (transformed)

- Location: `data/Platypus/theoremqa.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 747

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 747 | final answer only | theorem-application questions, answer-verified style |

_exact counts (full scan of the jsonl file)_

## Keyword: `Answer_type` (input)

Declared answer type per question (unused by the transform; shown for context).

| value | rows |
|---|---|
| `float` | 378 |
| `integer` | 216 |
| `bool` | 115 |
| `list of integer` | 61 |
| `option` | 18 |
| `list of float` | 12 |

_exact counts (all 800 rows)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `TIGER-Lab/TheoremQA`, split `test`

One row of HF `TIGER-Lab/TheoremQA`, split `test` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| Question | Answer | Picture |
|---|---|---|
| How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets? | 11760 | ∅ (null) |

### Transformed — condition=`direct`

One line of `data/Platypus/theoremqa.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?", "response": "11760"}
````

---

↑ [index](README.md) · ← [scienceqa](scienceqa.md) · [next → flan](flan.md)
