# webinstruct_verified

**Script:** `pipe/clean_webinstruct_verified.py`

## Purpose

Web-mined QA pairs whose answers were verified by LLM judges (TIGER-Lab/WebInstruct-verified). Broad-domain direct supervision.

## Before (raw storage)

- Source: HF `TIGER-Lab/WebInstruct-verified` (split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question` | string | the question |
| `answer` | string | verified answer |

## After (transformed)

- Location: `data/webinstruct_verified.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 228,736

`condition` values used here:

- `direct` — verified answer, no reasoning shown

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `TIGER-Lab/WebInstruct-verified`, split `train`)

````text
{
  "question": "Write a formula for the fourth degree polynomial p(x) whose graph is symmetric about the y-axis, which has a y-intercept of 10, and global maxima at (3,253) and (-3,253).",
  "answer": "y(x) = -3x^4 + 54x^2 + 10"
}
````

### Transformed record (`data/webinstruct_verified.jsonl`, record 1)

````text
{
  "condition": "direct",
  "instruction": "Write a formula for the fourth degree polynomial p(x) whose graph is symmetric about the y-axis, which has a y-intercept of 10, and global maxima at (3,253) and (-3,253).",
  "response": "y(x) = -3x^4 + 54x^2 + 10"
}
````

### Transformed record (`data/webinstruct_verified.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "You are buying a car and have borrowed $48,000 at an annual interest rate of 12 percent. The terms of the loan require you to make monthly payments and to completely amortize the loan over four years. Assuming you make the payments as agreed, what is the total amount of interest you will end up up paying over the four years?",
  "response": "$12,672.96"
}
````
