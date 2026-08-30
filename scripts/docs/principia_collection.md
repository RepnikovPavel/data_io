# principia_collection

**Script:** `pipe/clean_principia_collection.py`

## Purpose

Synthetic STEM problems generated from textbook/exam material (facebook/principia-collection). Straight question->answer pairs.

## Before (raw storage)

- Source: HF `facebook/principia-collection` (all splits)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `problem_statement` | string | the problem |
| `answer` | string | the answer |

## After (transformed)

- Location: `data/principia_collection.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 554,399

`condition` values used here:

- `synth,direct` — synthetic question->answer pair

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `facebook/principia-collection`)

````text
{
  "problem_statement": "Let the primitive logical connectives be negation (¬) and disjunction (∨) only.  \nUsing the standard truth‑functional definitions  \n\n\\[\nA\\land B\\;:=\\;\\neg(\\neg A\\;\\lor\\;\\neg B),\\qquad\nA\\to B\\;:=\\;\\neg A\\;\\lor\\;B,\n\\]\n\nprove that implication distributes over conjunction; that is, for arbitrary propositions \\(P\\), \\(Q\\) and \\(R\\),\n\n\\[\nP\\to (Q\\land R)\\;\\equiv\\;(P\\to Q)\\land(P\\to R).\n\\]\n\nYou must perform the proof **solely by algebraic manipulation of the expressions in terms of ¬ and ∨**, invoking only the basic Boolean identities (associativity, commutativity, distributivity of ∨ over ∧, double‑n\n… [truncated, 769 chars total]",
  "answer": "P\\to (Q\\land R)\\;\\equiv\\;(P\\to Q)\\land (P\\to R)"
}
````

### Transformed record (`data/principia_collection.jsonl`, record 1)

````text
{
  "condition": "synth,direct",
  "instruction": "Let the primitive logical connectives be negation (¬) and disjunction (∨) only.  \nUsing the standard truth‑functional definitions  \n\n\\[\nA\\land B\\;:=\\;\\neg(\\neg A\\;\\lor\\;\\neg B),\\qquad\nA\\to B\\;:=\\;\\neg A\\;\\lor\\;B,\n\\]\n\nprove that implication distributes over conjunction; that is, for arbitrary propositions \\(P\\), \\(Q\\) and \\(R\\),\n\n\\[\nP\\to (Q\\land R)\\;\\equiv\\;(P\\to Q)\\land(P\\to R).\n\\]\n\nYou must perform the proof **solely by algebraic manipulation of the expressions in terms of ¬ and ∨**, invoking only the basic Boolean identities (associativity, commutativity, distributivity of ∨ over ∧, double‑n\n… [truncated, 769 chars total]",
  "response": "P\\to (Q\\land R)\\;\\equiv\\;(P\\to Q)\\land (P\\to R)"
}
````

### Transformed record (`data/principia_collection.jsonl`, record 2)

````text
{
  "condition": "synth,direct",
  "instruction": "Write a single LaTeX‑compatible equation (containing an “=” sign) that *defines* semantic entailment of a set of formulas \\(\\Gamma\\) from a background theory \\(T\\) to a formula \\(\\varphi\\) in first‑order logic, denoted \\(\\Gamma \\models_T \\varphi\\). Your definition must quantify over **all** structures \\(\\mathcal{M}\\) (with the same signature) **and** over **all** variable assignments \\(s\\), using the satisfaction relation \\(\\mathcal{M},s \\models \\psi\\). Explicitly require that \\(\\mathcal{M}\\) be a model of the theory \\(T\\). The equation should equate the statement \\(\\Gamma \\models_T \\varphi\\) \n… [truncated, 800 chars total]",
  "response": "\\[\n\\Gamma \\models_T \\varphi \\;=\\; \\forall \\mathcal{M}\\,\\Bigl( \\mathcal{M}\\models T \\;\\rightarrow\\; \\forall s\\,\\bigl( \\bigl(\\forall \\psi\\in\\Gamma\\; \\mathcal{M},s\\models\\psi\\bigr) \\rightarrow \\mathcal{M},s\\models\\varphi \\bigr)\\Bigr)\n\\]"
}
````
