# amps_khan

↑ [index](README.md) · ← [webinstruct_verified](webinstruct_verified.md) · [next → arb](arb.md)

**Script:** `pipe/clean_amps_khan.py`

## Purpose

Khan Academy exercises from the AMPS dataset (Hendrycks et al.) — one JSON file per problem. The step hints are joined into the response; hint quality varies, hence noisy.

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/amps/khan/**/*.json` (extracted from `/mnt/hdd2/datasets_text/amps.tar.gz`)
- Storage: one JSON object per .json file, UTF-8

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `problem` | JSON string — UTF-8 text | the exercise text |
| `hints` | JSON array | step hints, joined with newlines into the response |

## After (transformed)

- Location: `data/amps_khan.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 103,059

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `noisy,cot` | 103,059 | hint sequence as pseudo-solution, quality unverified | cheap step-wise math signal of mixed quality |

_exact counts (full scan of the jsonl file)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/amps/khan/124/1607895295.json`

Full content of `/mnt/hdd2/datasets_text/amps/khan/124/1607895295.json` (one JSON object per file, UTF-8):

````json
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

### Transformed — condition=`noisy,cot`

One line of `data/amps_khan.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "noisy,cot", "instruction": "The first five terms of an arithmetic sequence are given: $4,10,16,22,28, \\ldots$ What is the sixth term in the sequence?", "response": "In any arithmetic sequence, each term is equal to the previous term plus the common difference.\nThus, the second term is equal to the first term plus the common difference. In this sequence, the second term,\n$10$\n, is\n$6$\nmore than the first term,\n$4$\nTherefore, the common difference is\n$6$\nThe sixth term in the sequence is equal to the fifth term plus the common difference, or $28 + 6 = 34$."}
````

---

↑ [index](README.md) · ← [webinstruct_verified](webinstruct_verified.md) · [next → arb](arb.md)
