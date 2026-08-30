# openbookqa

↑ [index](README.md) · ← [arb](arb.md) · [next → reclor](reclor.md)

**Script:** `pipe/clean_platypus/clean_openbookqa.py`

## Purpose

OpenBookQA: ~6k elementary science multiple-choice questions ('additional' config, which adds the supporting fact1). Question, options and fact are rendered into one instruction; the response is the option letter.

Created: 2018-09 · Domain: elementary science MCQ _(date source: OpenBookQA paper (EMNLP 2018), arXiv:1809.02789)_

## Before (raw storage)

- Source: HF `allenai/openbookqa` (config `additional`, splits `train`+`validation`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question_stem` | UTF-8 text (arrow `string`) | the question |
| `fact1` | UTF-8 text (arrow `string`) | supporting fact appended to the instruction |
| `choices` | arrow struct {text: arrow list — UTF-8 text (arrow `string`), label: arrow list — UTF-8 text (arrow `string`)} | answer options, rendered as A:/B:/C:/D: |
| `answerKey` | UTF-8 text (arrow `string`) | correct option letter (the response) |

## After (transformed)

- Location: `data/Platypus/openbookqa.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 5,457

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 5,457 | correct option letter | fact-grounded MCQ answering |

_exact counts (full scan of the jsonl file)_

## Keyword: `split` (input)

The two HF splits concatenated by the script (test is unused).

| value | rows |
|---|---|
| `train` | 4,957 |
| `validation` | 500 |

_exact (HF split sizes)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `allenai/openbookqa` (config `additional`), split `train`

One row of HF `allenai/openbookqa` (config `additional`), split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| question_stem | fact1 | choices | answerKey |
|---|---|---|---|
| The sun is responsible for | the sun is the source of energy for physical cycles on Earth | {"text": ["puppies learning new tricks", "children growing up and getting old", "flowers wilting in a vase", "plants sprouting, blooming and wilting"], "label": ["A", "B", "C", "D"]} | D |

### Transformed — condition=`direct`

One line of `data/Platypus/openbookqa.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "Based on the given fact, which of the following option is the correct answer to the question?\n\nThe sun is responsible for \nA: puppies learning new tricks\nB: children growing up and getting old\nC: flowers wilting in a vase\nD: plants sprouting, blooming and wilting\n\nFact: the sun is the source of energy for physical cycles on Earth", "response": "D"}
````

---

↑ [index](README.md) · ← [arb](arb.md) · [next → reclor](reclor.md)
