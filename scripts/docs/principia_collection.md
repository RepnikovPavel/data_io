# principia_collection

↑ [index](README.md) · ← [omnimath](omnimath.md) · [next → webinstruct_verified](webinstruct_verified.md)

**Script:** `pipe/clean_principia_collection.py`

## Purpose

Principia Collection (Meta): ~550k synthetic STEM problems (proposed by GPT-OSS-120B) over PhySH/MSC-2020 topics, in two splits: `mathematical_object` (derive an equation/expression) and `numerical` (numeric answer). Loaded verbatim as question->answer pairs.

Created: 2025-11 · Domain: synthetic STEM problems _(date source: HF release (repo createdAt 2025-11; paper pending per card))_

## Before (raw storage)

- Source: HF `facebook/principia-collection` (splits `mathematical_object`+`numerical`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `problem_statement` | UTF-8 text (arrow `string`) | the problem |
| `answer` | UTF-8 text (arrow `string`) | the answer |

## After (transformed)

- Location: `data/principia_collection.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 554,399

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `synth,direct` | 554,399 | synthetic question->answer pair | large-volume synthetic STEM drill data |

_exact counts (full scan of the jsonl file)_

## Keyword: `split` (input)

The two HF splits, concatenated in order by the script.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `numerical` | 305,656 | answer is numeric | teaches numeric problem solving |
| `mathematical_object` | 248,743 | answer is an equation/expression | teaches symbolic derivation |

_exact (HF split sizes)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `facebook/principia-collection`

One row of HF `facebook/principia-collection` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| problem_statement | answer |
|---|---|
| Let the primitive logical connectives be negation (¬) and disjunction (∨) only.  ⏎Using the standard truth‑functional definitions  ⏎⏎\[⏎A\land B\;:=\;\neg(\neg A\;\lor\;\neg B),\qquad⏎A\to B\;:=\;\neg A\;\lor\;B,⏎\]⏎⏎prove that implication distributes over conjunction; that is, for arbitrary propositions \(P\), \(Q\) and \(R\),⏎⏎\[⏎P\to (Q\land R)\;\equiv\;(P\to Q)\land(P\to R).⏎\]⏎⏎You must perform the proof **solely by algebraic manipulation of the expressions in terms of ¬ and ∨**, invoking only the basic Boolean identities (associativity, commutativity, distributivity of ∨ over ∧, double‑n… [truncated, 769 chars total] | P\to (Q\land R)\;\equiv\;(P\to Q)\land (P\to R) |

### Transformed — condition=`synth,direct`

One line of `data/principia_collection.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "synth,direct", "instruction": "Let the primitive logical connectives be negation (¬) and disjunction (∨) only.  \nUsing the standard truth‑functional definitions  \n\n\\[\nA\\land B\\;:=\\;\\neg(\\neg A\\;\\lor\\;\\neg B),\\qquad\nA\\to B\\;:=\\;\\neg A\\;\\lor\\;B,\n\\]\n\nprove that implication distributes over conjunction; that is, for arbitrary propositions \\(P\\), \\(Q\\) and \\(R\\),\n\n\\[\nP\\to (Q\\land R)\\;\\equiv\\;(P\\to Q)\\land(P\\to R).\n\\]\n\nYou must perform the proof **solely by algebraic manipulation of the expressions in terms of ¬ and ∨**, invoking only the basic Boolean identities (associativity, commutativity, distributivity of ∨ over ∧, double‑n… [truncated, 769 chars total]", "response": "P\\to (Q\\land R)\\;\\equiv\\;(P\\to Q)\\land (P\\to R)"}
````

---

↑ [index](README.md) · ← [omnimath](omnimath.md) · [next → webinstruct_verified](webinstruct_verified.md)
