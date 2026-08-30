# arb

↑ [index](README.md) · ← [amps_khan](amps_khan.md) · [next → openbookqa](openbookqa.md)

**Script:** `pipe/clean_platypus/clean_arb.py`

## Purpose

ARB: an advanced reasoning benchmark with graduate-level problems in math, physics, science, reading and law (distributed with Platypus, one JSON file per subject). The transform prepends a fixed task description per subject and tags law as direct (option letter), the rest as cot.

Created: 2023-07 · Domain: advanced reasoning (graduate STEM/law/reading) _(date source: ARB paper, arXiv:2307.13692)_

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/Platypus/ARB/*.json` (5 subject files)
- Storage: one JSON array of objects per subject file, UTF-8

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `instruction` | JSON string — UTF-8 text | the problem (a subject-specific description is prepended) |
| `response` | JSON string — UTF-8 text | the solution |

## After (transformed)

- Location: `data/Platypus/arb_*.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 5 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 901

| column | type | meaning |
|---|---|---|
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 627 | law: correct option letter only | MCQ-style answer-only supervision |
| `cot` | 274 | math / reading / science / physics: worked solution | advanced graduate-level reasoning |

_exact counts (full scan of the jsonl file)_

## Keyword: `subject file` (input)

One input JSON file per subject; the file determines the prepended instruction prefix and the condition tag.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `law` | 627 | 'Choose the correct option letter.' | tagged direct |
| `physics` | 80 | 'Solve the physics problem.' | tagged cot |
| `reading` | 77 | 'Solve the reading comprehension problem.' | tagged cot |
| `science` | 65 | 'Solve the science problem.' | tagged cot |
| `math` | 52 | 'Solve the math problem.' | tagged cot |

_exact (rows summed per file from metadata/line counts)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/Platypus/ARB/math.json`

First element of the JSON array in `/mnt/hdd2/datasets_text/Platypus/ARB/math.json` (UTF-8 JSON file):

````json
{
  "instruction": "Compute$$ \\int_{0}^{\\pi} \\frac{x \\sin x}{1+\\sin ^{2} x} d x . $$",
  "response": "We use the example from the introduction for the particular function $f(x)=\\frac{x}{1+x^{2}}$ to transform the integral into$$ \\pi \\int_{0}^{\\frac{\\pi}{2}} \\frac{\\sin x}{1+\\sin ^{2} x} d x . $$This is the same as$$ \\pi \\int_{0}^{\\frac{\\pi}{2}}-\\frac{d(\\cos x)}{2-\\cos ^{2} x}, $$which with the substitution $t=\\cos x$ becomes$$ \\pi \\int_{0}^{1} \\frac{1}{2-t^{2}} d t=\\left.\\frac{\\pi}{2 \\sqrt{2}} \\ln \\frac{\\sqrt{2}+t}{\\sqrt{2}-t}\\right|_{0} ^{1}=\\frac{\\pi}{2 \\sqrt{2}} \\ln \\frac{\\sqrt{2}+1}{\\sqrt{2}-1} . $$"
}
````

### Transformed — condition=`direct`

One line of `data/Platypus/arb_law.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"instruction": "Choose the correct option letter.\n\nFor a number of years, United Leasing has been in charge of leasing the luxury skyboxes at City Sports Stadium, home of the local professional basketball team. During this time, it annually sent to chief executives of area businesses personalized \"invitations\" to lease skyboxes for the season. The invitations, which were always sent out several months before each season began, contained detailed price terms and language stating that the deadline for responding was 10 weeks before the start of the season and that all leases were subject to the approval of the … [truncated, 2607 chars total]", "response": "A", "condition": "direct"}
````

### Transformed — condition=`cot`

One line of `data/Platypus/arb_math.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"instruction": "Solve the math problem.\n\nCompute$$ \\int_{0}^{\\pi} \\frac{x \\sin x}{1+\\sin ^{2} x} d x . $$", "response": "We use the example from the introduction for the particular function $f(x)=\\frac{x}{1+x^{2}}$ to transform the integral into$$ \\pi \\int_{0}^{\\frac{\\pi}{2}} \\frac{\\sin x}{1+\\sin ^{2} x} d x . $$This is the same as$$ \\pi \\int_{0}^{\\frac{\\pi}{2}}-\\frac{d(\\cos x)}{2-\\cos ^{2} x}, $$which with the substitution $t=\\cos x$ becomes$$ \\pi \\int_{0}^{1} \\frac{1}{2-t^{2}} d t=\\left.\\frac{\\pi}{2 \\sqrt{2}} \\ln \\frac{\\sqrt{2}+t}{\\sqrt{2}-t}\\right|_{0} ^{1}=\\frac{\\pi}{2 \\sqrt{2}} \\ln \\frac{\\sqrt{2}+1}{\\sqrt{2}-1} . $$", "condition": "cot"}
````

---

↑ [index](README.md) · ← [amps_khan](amps_khan.md) · [next → openbookqa](openbookqa.md)
