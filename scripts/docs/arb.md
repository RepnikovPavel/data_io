# arb

**Script:** `pipe/clean_platypus/clean_arb.py`

## Purpose

ARB advanced reasoning benchmark (math, physics, science, reading, law) as distributed with Platypus. A fixed task description is prepended to each problem depending on the subject file.

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/Platypus/ARB/*.json` (5 subject files)
- Format: JSON list of objects per subject file

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `instruction` | string | the problem (a subject-specific description is prepended) |
| `response` | string | the solution |

## After (transformed)

- Location: `data/Platypus/arb_*.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 5 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 901

`condition` values used here:

- `cot` — math / reading / science / physics: worked solution
- `direct` — law: correct option letter only

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/Platypus/ARB/math.json`)

````text
{
  "instruction": "Compute$$ \\int_{0}^{\\pi} \\frac{x \\sin x}{1+\\sin ^{2} x} d x . $$",
  "response": "We use the example from the introduction for the particular function $f(x)=\\frac{x}{1+x^{2}}$ to transform the integral into$$ \\pi \\int_{0}^{\\frac{\\pi}{2}} \\frac{\\sin x}{1+\\sin ^{2} x} d x . $$This is the same as$$ \\pi \\int_{0}^{\\frac{\\pi}{2}}-\\frac{d(\\cos x)}{2-\\cos ^{2} x}, $$which with the substitution $t=\\cos x$ becomes$$ \\pi \\int_{0}^{1} \\frac{1}{2-t^{2}} d t=\\left.\\frac{\\pi}{2 \\sqrt{2}} \\ln \\frac{\\sqrt{2}+t}{\\sqrt{2}-t}\\right|_{0} ^{1}=\\frac{\\pi}{2 \\sqrt{2}} \\ln \\frac{\\sqrt{2}+1}{\\sqrt{2}-1} . $$"
}
````

### Transformed record (`data/Platypus/arb_law.jsonl`, record 1)

````text
{
  "instruction": "Choose the correct option letter.\n\nFor a number of years, United Leasing has been in charge of leasing the luxury skyboxes at City Sports Stadium, home of the local professional basketball team. During this time, it annually sent to chief executives of area businesses personalized \"invitations\" to lease skyboxes for the season. The invitations, which were always sent out several months before each season began, contained detailed price terms and language stating that the deadline for responding was 10 weeks before the start of the season and that all leases were subject to the approval of the \n… [truncated, 2607 chars total]",
  "response": "A",
  "condition": "direct"
}
````

### Transformed record (`data/Platypus/arb_law.jsonl`, record 2)

````text
{
  "instruction": "Choose the correct option letter.\n\nBulky was six foot four and weighed 280 pounds. One afternoon Bulky was wandering rather aimlessly and became lost in an unfamiliar part of the city. He reached into his pocket and discovered he only had 35 cents. He wanted to take a bus back to the city center, but bus fare was $ 1$ per ride. Bulky was rather scruffily dressed and he had not had a haircut in several months. He approached Juan, a slightly built man who was standing alone at the bus stop. In a gruff voice, Bulky asked Juan, \"Do you have any money?\" When Juan replied, \"Yes,\"'Bulky said, \"Give m\n… [truncated, 1603 chars total]",
  "response": "A",
  "condition": "direct"
}
````
