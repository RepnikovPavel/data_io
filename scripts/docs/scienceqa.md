# scienceqa

**Script:** `pipe/clean_platypus/clean_scienceqa.py`

## Purpose

ScienceQA text-only multiple-choice science questions. Rows with a rationale yield a cot record (rationale + answer letter); every row also yields a direct record (bare letter). A lecture, when present, is appended to the instruction.

## Before (raw storage)

- Source: HF `metaeval/ScienceQA_text_only` (splits `train`+`validation`+`test`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question` | string | the question |
| `choices` | list[string] | options, rendered as A:/B:/C:/... |
| `lecture` | string | background text, appended to the instruction when non-empty |
| `solution` | string | rationale; when empty no cot record is emitted |
| `answer` | int | index of the correct option (response is the letter) |

## After (transformed)

- Location: `data/Platypus/scienceqa.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 20,642

`condition` values used here:

- `cot` — rationale + 'Answer: X'
- `direct` — bare option letter

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `metaeval/ScienceQA_text_only`, split `train`)

````text
{
  "question": "Which tense does the sentence use?\nMona will print her name with care.",
  "choices": [
    "present tense",
    "future tense",
    "past tense"
  ],
  "lecture": "Present tense verbs tell you about something that is happening now.\nMost present-tense verbs are regular. They have no ending, or they end in -s or -es.\nTwo verbs are irregular in the present tense, to be and to have. You must remember their forms.\nPast tense verbs tell you about something that has already happened.\nMost past-tense verbs are regular. They end in -ed.\nSome verbs are irregular in the past tense. You must remember their past-tense forms.\nFuture tense verbs tell you about something that is going to happen.\nAll future-tense verbs use the word will.\nPresent | Past | Future\nwalk, wal\n… [truncated, 649 chars total]",
  "solution": "The sentence is in future tense. You can tell because it uses will before the main verb, print. The verb tells you about something that is going to happen.",
  "answer": 1
}
````

### Transformed record (`data/Platypus/scienceqa.jsonl`, record 1)

````text
{
  "condition": "cot",
  "instruction": "Solve the following question using the information provided in the lecture.\n\nWhich tense does the sentence use?\nMona will print her name with care.\nOptions:\nA: present tense\nB: future tense\nC: past tense\n\nLecture: Present tense verbs tell you about something that is happening now.\nMost present-tense verbs are regular. They have no ending, or they end in -s or -es.\nTwo verbs are irregular in the present tense, to be and to have. You must remember their forms.\nPast tense verbs tell you about something that has already happened.\nMost past-tense verbs are regular. They end in -ed.\nSome verbs are i\n… [truncated, 863 chars total]",
  "response": "The sentence is in future tense. You can tell because it uses will before the main verb, print. The verb tells you about something that is going to happen.\n\nAnswer: B"
}
````

### Transformed record (`data/Platypus/scienceqa.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "Choose the correct option letter for the following question based on the information from the lecture.\n\nWhich tense does the sentence use?\nMona will print her name with care.\nOptions:\nA: present tense\nB: future tense\nC: past tense\n\nLecture: Present tense verbs tell you about something that is happening now.\nMost present-tense verbs are regular. They have no ending, or they end in -s or -es.\nTwo verbs are irregular in the present tense, to be and to have. You must remember their forms.\nPast tense verbs tell you about something that has already happened.\nMost past-tense verbs are regular. They e\n… [truncated, 890 chars total]",
  "response": "B"
}
````
