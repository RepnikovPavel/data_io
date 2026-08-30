# amps_khan

**Script:** `pipe/clean_amps_khan.py`

## Purpose

Khan Academy exercises from the AMPS dataset (one JSON file per problem). The step hints are joined into the response — hint quality varies, hence noisy.

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/amps/khan/**/*.json` (extracted from `/mnt/hdd2/datasets_text/amps.tar.gz`)
- Format: one JSON object per file

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `problem` | string | the exercise text |
| `hints` | list[string] | step hints, joined with newlines into the response |

## After (transformed)

- Location: `data/amps_khan.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 103,059

`condition` values used here:

- `noisy,cot` — hint sequence as pseudo-solution, quality unverified

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/amps/khan/124/1607895295.json`)

````text
{
  "problem": "The first five terms of an arithmetic sequence are given: $4,10,16,22,28, \\ldots$ What is the sixth term in the sequence?",
  "hints": [
    "In any arithmetic sequence, each term is equal to the previous term plus the common difference.",
    "Thus, the second term is equal to the first term plus the common difference. In this sequence, the second term,",
    "$10$",
    ", is",
    "$6$",
    "more than the first term,",
    "$4$",
    "Therefore, the common difference is",
    "$6$",
    "The sixth term in the sequence is equal to the fifth term plus the common difference, or $28 + 6 = 34$."
  ]
}
````

### Transformed record (`data/amps_khan.jsonl`, record 1)

````text
{
  "condition": "noisy,cot",
  "instruction": "The first five terms of an arithmetic sequence are given: $4,10,16,22,28, \\ldots$ What is the sixth term in the sequence?",
  "response": "In any arithmetic sequence, each term is equal to the previous term plus the common difference.\nThus, the second term is equal to the first term plus the common difference. In this sequence, the second term,\n$10$\n, is\n$6$\nmore than the first term,\n$4$\nTherefore, the common difference is\n$6$\nThe sixth term in the sequence is equal to the fifth term plus the common difference, or $28 + 6 = 34$."
}
````

### Transformed record (`data/amps_khan.jsonl`, record 2)

````text
{
  "condition": "noisy,cot",
  "instruction": "The first four terms of an arithmetic sequence are given: $-1,2,5,8, \\ldots$ What is the fifth term in the sequence?",
  "response": "In any arithmetic sequence, each term is equal to the previous term plus the common difference.\nThus, the second term is equal to the first term plus the common difference. In this sequence, the second term,\n$2$\n, is\n$3$\nmore than the first term,\n$-1$\nTherefore, the common difference is\n$3$\nThe fifth term in the sequence is equal to the fourth term plus the common difference, or $8 + 3 = 11$."
}
````
