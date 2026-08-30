# webinstruct_verified

↑ [index](README.md) · ← [principia_collection](principia_collection.md) · [next → amps_khan](amps_khan.md)

**Script:** `pipe/clean_webinstruct_verified.py`

## Purpose

WebInstruct-verified (TIGER-Lab, General-Reasoner project): ~230k web-mined questions across many domains whose answers were verified for correctness/verifiability. Loaded verbatim as direct QA pairs.

## Before (raw storage)

- Source: HF `TIGER-Lab/WebInstruct-verified` (split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question` | UTF-8 text (arrow `string`) | the question |
| `answer` | UTF-8 text (arrow `string`) | verified answer |

## After (transformed)

- Location: `data/webinstruct_verified.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 228,736

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 228,736 | verified answer, no reasoning shown | broad-domain verifiable QA supervision |

_exact counts (full scan of the jsonl file)_

## Keyword: `category` (input)

Domain of the question (unused by the transform; shown for context).

| value | rows |
|---|---|
| `Mathematics` | 10,206 |
| `Physics` | 7,086 |
| `Chemistry` | 3,243 |
| `Business` | 3,025 |
| `Finance` | 1,747 |
| `Economics` | 1,540 |
| `Other` | 894 |
| `History` | 780 |
| `Biology` | 712 |
| `Psychology` | 183 |
| `Computer Science` | 169 |
| `Health` | 135 |
| `Other STEM` | 112 |
| `Philosophy` | 79 |
| `Law` | 56 |
| `Engineering` | 33 |

_estimated from the first 30,000 of 228,736 rows_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `TIGER-Lab/WebInstruct-verified`, split `train`

One row of HF `TIGER-Lab/WebInstruct-verified`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| question | answer |
|---|---|
| Write a formula for the fourth degree polynomial p(x) whose graph is symmetric about the y-axis, which has a y-intercept of 10, and global maxima at (3,253) and (-3,253). | y(x) = -3x^4 + 54x^2 + 10 |

### Transformed — condition=`direct`

One line of `data/webinstruct_verified.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "Write a formula for the fourth degree polynomial p(x) whose graph is symmetric about the y-axis, which has a y-intercept of 10, and global maxima at (3,253) and (-3,253).", "response": "y(x) = -3x^4 + 54x^2 + 10"}
````

---

↑ [index](README.md) · ← [principia_collection](principia_collection.md) · [next → amps_khan](amps_khan.md)
