# reclor

**Script:** `pipe/clean_platypus/clean_reclor.py`

## Purpose

ReClor logical reasoning multiple-choice questions (LSAT/GMAT style). Context, question and options are rendered into one instruction.

## Before (raw storage)

- Source: HF `metaeval/reclor` (splits `train`+`validation`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `context` | string | passage the question refers to |
| `question` | string | the question |
| `answers` | list[string] | options, rendered as A:/B:/C:/D: |
| `label` | int | index of the correct option (response is the letter) |

## After (transformed)

- Location: `data/Platypus/reclor.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 5,138

`condition` values used here:

- `direct` — correct option letter

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `metaeval/reclor`, split `train`)

````text
{
  "context": "In rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.",
  "question": "The statements above, if true, most strongly support which one of the following conclusions?",
  "answers": [
    "Unlike aspirin and other medications that reduce pain and swelling and that are currently available, the new medication would repair existing cell damage that had been caused by rheumatoid arthritis.",
    "A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.",
    "Joint diseases other than rheumatoid arthritis would not be affected by the new medication.",
    "The benefits to rheumatoid arthritis sufferers of the new medication would outweigh the medication's possible harmful side effects."
  ],
  "label": 1
}
````

### Transformed record (`data/Platypus/reclor.jsonl`, record 1)

````text
{
  "condition": "direct",
  "instruction": "The statements above, if true, most strongly support which one of the following conclusions?\n\nIn rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.\n\nOptions:\nA: Unlike aspirin and other medications that reduce pain and swelling and that are currently available,\n… [truncated, 1058 chars total]",
  "response": "B"
}
````

### Transformed record (`data/Platypus/reclor.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "The patient's argument proceeds by\n\nPatient: Pharmacists maintain that doctors should not be permitted to sell the medicine that they prescribe because doctors would then be tempted to prescribe unnecessary medicines in order to earn extra income. But pharmacists have a financial interest in having a monopoly on the sale of prescription medicines, so their objection to the sale of medicines by doctors cannot be taken seriously.\n\nOptions:\nA: attempting to discredit a position by questioning the motives of the proponents of that position\nB: rejecting a questionable position on the grounds that t\n… [truncated, 869 chars total]",
  "response": "A"
}
````
