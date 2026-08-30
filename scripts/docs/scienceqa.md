# scienceqa

↑ [index](README.md) · ← [scibench](scibench.md) · [next → theoremqa](theoremqa.md)

**Script:** `pipe/clean_platypus/clean_scienceqa.py`

## Purpose

ScienceQA (text-only subset): grade-school science MCQs with lectures and rationales. Rows with a rationale yield a cot record (rationale + answer letter); every row also yields a direct record (bare letter). A lecture, when present, is appended to the instruction.

Created: 2022-09 · Domain: grade-school science MCQ _(date source: ScienceQA paper (NeurIPS 2022), arXiv:2209.09513)_

## Before (raw storage)

- Source: HF `metaeval/ScienceQA_text_only` (splits `train`+`validation`+`test`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question` | UTF-8 text (arrow `string`) | the question |
| `choices` | arrow list — UTF-8 text (arrow `string`) | options, rendered as A:/B:/C:/... |
| `lecture` | UTF-8 text (arrow `string`) | background text, appended to the instruction when non-empty |
| `solution` | UTF-8 text (arrow `string`) | rationale; when empty no cot record is emitted |
| `answer` | integer (arrow `int8`) | index of the correct option (response is the letter) |

## After (transformed)

- Location: `data/Platypus/scienceqa.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 20,642

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 10,876 | bare option letter | answer-only variant of every question |
| `cot` | 9,766 | rationale + 'Answer: X' | teaches explaining the answer before giving it |

_exact counts (full scan of the jsonl file)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `metaeval/ScienceQA_text_only`, split `train`

One row of HF `metaeval/ScienceQA_text_only`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| question | choices | lecture | solution | answer |
|---|---|---|---|---|
| Which tense does the sentence use?⏎Mona will print her name with care. | ["present tense", "future tense", "past tense"] | Present tense verbs tell you about something that is happening now.⏎Most present-tense verbs are regular. They have no ending, or they end in -s or -es.⏎Two verbs are irregular in the present tense, to be and to have. You must remember their forms.⏎Past tense verbs tell you about something that has already happened.⏎Most past-tense verbs are regular. They end in -ed.⏎Some verbs are irregular in the past tense. You must remember their past-tense forms.⏎Future tense verbs tell you about something that is going to happen.⏎All future-tense verbs use the word will.⏎Present \| Past \| Future⏎walk, wal… [truncated, 649 chars total] | The sentence is in future tense. You can tell because it uses will before the main verb, print. The verb tells you about something that is going to happen. | 1 |

### Transformed — condition=`cot`

One line of `data/Platypus/scienceqa.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "cot", "instruction": "Solve the following question using the information provided in the lecture.\n\nWhich tense does the sentence use?\nMona will print her name with care.\nOptions:\nA: present tense\nB: future tense\nC: past tense\n\nLecture: Present tense verbs tell you about something that is happening now.\nMost present-tense verbs are regular. They have no ending, or they end in -s or -es.\nTwo verbs are irregular in the present tense, to be and to have. You must remember their forms.\nPast tense verbs tell you about something that has already happened.\nMost past-tense verbs are regular. They end in -ed.\nSome verbs are i… [truncated, 863 chars total]", "response": "The sentence is in future tense. You can tell because it uses will before the main verb, print. The verb tells you about something that is going to happen.\n\nAnswer: B"}
````

### Transformed — condition=`direct`

One line of `data/Platypus/scienceqa.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "Choose the correct option letter for the following question based on the information from the lecture.\n\nWhich tense does the sentence use?\nMona will print her name with care.\nOptions:\nA: present tense\nB: future tense\nC: past tense\n\nLecture: Present tense verbs tell you about something that is happening now.\nMost present-tense verbs are regular. They have no ending, or they end in -s or -es.\nTwo verbs are irregular in the present tense, to be and to have. You must remember their forms.\nPast tense verbs tell you about something that has already happened.\nMost past-tense verbs are regular. They e… [truncated, 890 chars total]", "response": "B"}
````

---

↑ [index](README.md) · ← [scibench](scibench.md) · [next → theoremqa](theoremqa.md)
