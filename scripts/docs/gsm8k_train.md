# gsm8k_train

↑ [index](README.md) · [next → math_train](math_train.md)

**Script:** `pipe/clean_gsm8k_train.py`

## Purpose

GSM8K (OpenAI): 7,473 crowdsourced grade-school math word problems (train split; the socratic config is unused). Raw answers embed calculator annotations `<<...>>` and end in `#### <final>`; the transform keeps only the final answer, teaching terse numeric answers to word problems.

## Before (raw storage)

- Source: HF `openai/gsm8k` (config `main`, split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question` | UTF-8 text (arrow `string`) | the word problem |
| `answer` | UTF-8 text (arrow `string`) | worked solution ending in `#### <final>`; only the part after `####` is kept |

## After (transformed)

- Location: `data/gsm8k_train.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 7,473

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 7,473 | response is the final numeric answer only | pure answer-only supervision for easy arithmetic |

_exact counts (full scan of the jsonl file)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `openai/gsm8k` (config `main`), split `train`

One row of HF `openai/gsm8k` (config `main`), split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| question | answer |
|---|---|
| Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? | Natalia sold 48/2 = <<48/2=24>>24 clips in May.⏎Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.⏎#### 72 |

### Transformed — condition=`direct`

One line of `data/gsm8k_train.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "response": "72"}
````

---

↑ [index](README.md) · [next → math_train](math_train.md)
