# natural_reasoning

**Script:** `pipe/clean_natural_reasoning.py`

## Purpose

General-knowledge reasoning questions with reference answers (facebook/natural_reasoning). Rows asking to 'prove'/'show that' and empty answers are filtered out.

## Before (raw storage)

- Source: HF `facebook/natural_reasoning` (split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question` | string | the question |
| `reference_answer` | string | reference answer (stripped; empty -> dropped) |

## After (transformed)

- Location: `data/natural_reasoning.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 770,141

`condition` values used here:

- `noisy,direct` — reference answer as-is; quality not verified

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `facebook/natural_reasoning`, split `train`)

````text
{
  "question": "What is the total work done on an object when it is moved upwards against gravity, considering both the change in kinetic energy and potential energy? Use the Work-Energy Theorem and the principle of conservation of mechanical energy to derive your answer.",
  "reference_answer": "W = delta ME = delta KE + delta PE"
}
````

### Transformed record (`data/natural_reasoning.jsonl`, record 1)

````text
{
  "condition": "noisy,direct",
  "instruction": "What is the total work done on an object when it is moved upwards against gravity, considering both the change in kinetic energy and potential energy? Use the Work-Energy Theorem and the principle of conservation of mechanical energy to derive your answer.",
  "response": "W = delta ME = delta KE + delta PE"
}
````

### Transformed record (`data/natural_reasoning.jsonl`, record 2)

````text
{
  "condition": "noisy,direct",
  "instruction": "Propose a system of 'Practical Numbers' that denies the Axiom of Choice and the notion of infinity. Discuss how such a system could be constructed, considering the implications for set theory and the foundations of mathematics. How might the usual results in analysis be affected, and what potential benefits or drawbacks could this system have for mathematical modeling and physics?",
  "response": "A well-structured proposal addressing the challenges and implications of constructing a system without infinity, including discussions on set theory, analysis, and potential applications."
}
````
