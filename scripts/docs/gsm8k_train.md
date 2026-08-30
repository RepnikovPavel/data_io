# gsm8k_train

**Script:** `pipe/clean_gsm8k_train.py`

## Purpose

Grade-school math word problems (GSM8K train split). Teaches short arithmetic problem solving with a bare final numeric answer — the annotated calculator steps are discarded.

## Before (raw storage)

- Source: HF `openai/gsm8k` (config `main`, split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question` | string | the word problem |
| `answer` | string | worked solution ending in `#### <final>`; only the part after `####` is kept |

## After (transformed)

- Location: `data/gsm8k_train.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 7,473

`condition` values used here:

- `direct` — response is the final numeric answer only

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `openai/gsm8k` (config `main`), split `train`)

````text
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
````

### Transformed record (`data/gsm8k_train.jsonl`, record 1)

````text
{
  "condition": "direct",
  "instruction": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "response": "72"
}
````

### Transformed record (`data/gsm8k_train.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
  "response": "10"
}
````
