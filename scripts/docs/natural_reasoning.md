# natural_reasoning

↑ [index](README.md) · ← [math_train](math_train.md) · [next → no_robots](no_robots.md)

**Script:** `pipe/clean_natural_reasoning.py`

## Purpose

NaturalReasoning (Meta): 1.1M challenging reasoning questions backtranslated from DCLM/FineMath pretraining corpora, deduplicated and decontaminated against MATH/GPQA/MMLU. The transform keeps the reference answer extracted from the source document (rows with empty answers or proof-style questions are dropped).

Created: 2025-02 · Domain: web-derived reasoning QA _(date source: NaturalReasoning paper, arXiv:2502.13124)_

## Before (raw storage)

- Source: HF `facebook/natural_reasoning` (split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question` | UTF-8 text (arrow `string`) | the question |
| `reference_answer` | UTF-8 text (arrow `string`) | reference answer from the source document (stripped; empty -> dropped) |

## After (transformed)

- Location: `data/natural_reasoning.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 770,141

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `noisy,direct` | 770,141 | reference answer as-is, unverified | broad reasoning QA whose answers were not model-verified |

_exact counts (full scan of the jsonl file)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `facebook/natural_reasoning`, split `train`

One row of HF `facebook/natural_reasoning`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| question | reference_answer |
|---|---|
| What is the total work done on an object when it is moved upwards against gravity, considering both the change in kinetic energy and potential energy? Use the Work-Energy Theorem and the principle of conservation of mechanical energy to derive your answer. | W = delta ME = delta KE + delta PE |

### Transformed — condition=`noisy,direct`

One line of `data/natural_reasoning.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "noisy,direct", "instruction": "What is the total work done on an object when it is moved upwards against gravity, considering both the change in kinetic energy and potential energy? Use the Work-Energy Theorem and the principle of conservation of mechanical energy to derive your answer.", "response": "W = delta ME = delta KE + delta PE"}
````

---

↑ [index](README.md) · ← [math_train](math_train.md) · [next → no_robots](no_robots.md)
