# reclor

↑ [index](README.md) · ← [openbookqa](openbookqa.md) · [next → scibench](scibench.md)

**Script:** `pipe/clean_platypus/clean_reclor.py`

## Purpose

ReClor: logical-reasoning reading comprehension MCQs from LSAT/GMAT preparation material. Context, question and options are rendered into one instruction; the response is the option letter.

Created: 2020-02 · Domain: logical reasoning MCQ _(date source: ReClor paper (ICLR 2020), arXiv:2002.04326)_

## Before (raw storage)

- Source: HF `metaeval/reclor` (splits `train`+`validation`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `context` | UTF-8 text (arrow `string`) | passage the question refers to |
| `question` | UTF-8 text (arrow `string`) | the question |
| `answers` | arrow list — UTF-8 text (arrow `string`) | options, rendered as A:/B:/C:/D: |
| `label` | integer (arrow `int64`) | index of the correct option (response is the letter) |

## After (transformed)

- Location: `data/Platypus/reclor.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 5,138

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 5,138 | correct option letter | logical-reasoning MCQ answering |

_exact counts (full scan of the jsonl file)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `metaeval/reclor`, split `train`

One row of HF `metaeval/reclor`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| context | question | answers | label |
|---|---|---|---|
| In rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints. | The statements above, if true, most strongly support which one of the following conclusions? | ["Unlike aspirin and other medications that reduce pain and swelling and that are currently available, the new medication would repair existing cell damage that had been caused by rheumatoid arthritis.", "A patient treated with the new medication for rheumatoid arthritis could sustain a joint injury without becoming aware of it.", "Joint diseases other than rheumatoid arthritis would not be affected by the new medication.", "The benefits to rheumatoid arthritis sufferers of the new medication would outweigh the medication's possible harmful side effects."] | 1 |

### Transformed — condition=`direct`

One line of `data/Platypus/reclor.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "The statements above, if true, most strongly support which one of the following conclusions?\n\nIn rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints.\n\nOptions:\nA: Unlike aspirin and other medications that reduce pain and swelling and that are currently available,… [truncated, 1058 chars total]", "response": "B"}
````

---

↑ [index](README.md) · ← [openbookqa](openbookqa.md) · [next → scibench](scibench.md)
