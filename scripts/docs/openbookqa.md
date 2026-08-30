# openbookqa

**Script:** `pipe/clean_platypus/clean_openbookqa.py`

## Purpose

OpenBookQA elementary science multiple-choice questions ('additional' config with the supporting fact1). Question, options and fact are rendered into one instruction.

## Before (raw storage)

- Source: HF `allenai/openbookqa` (config `additional`, splits `train`+`validation`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question_stem` | string | the question |
| `fact1` | string | supporting fact appended to the instruction |
| `choices` | {text: list[string], label: list[string]} | answer options, rendered as A:/B:/C:/D: |
| `answerKey` | string | correct option letter (the response) |

## After (transformed)

- Location: `data/Platypus/openbookqa.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 5,457

`condition` values used here:

- `direct` — correct option letter

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `allenai/openbookqa` (config `additional`), split `train`)

````text
{
  "question_stem": "The sun is responsible for",
  "fact1": "the sun is the source of energy for physical cycles on Earth",
  "choices": {
    "text": [
      "puppies learning new tricks",
      "children growing up and getting old",
      "flowers wilting in a vase",
      "plants sprouting, blooming and wilting"
    ],
    "label": [
      "A",
      "B",
      "C",
      "D"
    ]
  },
  "answerKey": "D"
}
````

### Transformed record (`data/Platypus/openbookqa.jsonl`, record 1)

````text
{
  "condition": "direct",
  "instruction": "Based on the given fact, which of the following option is the correct answer to the question?\n\nThe sun is responsible for \nA: puppies learning new tricks\nB: children growing up and getting old\nC: flowers wilting in a vase\nD: plants sprouting, blooming and wilting\n\nFact: the sun is the source of energy for physical cycles on Earth",
  "response": "D"
}
````

### Transformed record (`data/Platypus/openbookqa.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "Based on the given fact, which of the following option is the correct answer to the question?\n\nWhen standing miles away from Mount Rushmore \nA: the mountains seem very close\nB: the mountains are boring\nC: the mountains look the same as from up close\nD: the mountains seem smaller than in photographs\n\nFact: as distance to an object increases , that object will appear smaller",
  "response": "D"
}
````
