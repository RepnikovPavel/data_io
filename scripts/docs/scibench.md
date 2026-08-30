# scibench

**Script:** `pipe/clean_platypus/clean_scibench.py`

## Purpose

SciBench college-level scientific problems (per-textbook JSON files). Emits the worked solution when present (cot) and always the final numeric answer (direct).

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/Platypus/scibench/dataset/original/*.json`
- Format: JSON list of objects per textbook file

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `problem_text` | string | the problem |
| `solution` | string | worked solution (cot record emitted only when non-empty) |
| `answer_latex` | string | final answer in LaTeX (falls back to answer_number) |
| `answer_number` | string | numeric final answer; used when answer_latex is redundant |

## After (transformed)

- Location: `data/Platypus/scibench.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 867

`condition` values used here:

- `cot` — worked solution (only when the source has one)
- `direct` — final answer (answer_latex, simplified when it duplicates answer_number)

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/Platypus/scibench/dataset/original/atkins.json`)

````text
{
  "problem_text": "Suppose that $10.0 \\mathrm{~mol} \\mathrm{C}_2 \\mathrm{H}_6(\\mathrm{~g})$ is confined to $4.860 \\mathrm{dm}^3$ at $27^{\\circ} \\mathrm{C}$. Predict the pressure exerted by the ethane from the perfect gas.",
  "solution": null,
  "answer_latex": " 50.7",
  "answer_number": "50.7"
}
````

### Transformed record (`data/Platypus/scibench.jsonl`, record 1)

````text
{
  "condition": "cot",
  "instruction": "For instance, suppose that one opens an individual retirement account (IRA) at age 25 and makes annual investments of $\\$ 2000$ thereafter in a continuous manner. Assuming a rate of return of $8 \\%$, what will be the balance in the IRA at age 65 ?",
  "response": "We have $S_0=0, r=0.08$, and $k=\\$ 2000$, and we wish to determine $S(40)$. From Eq. $$\nS(t)=S_0 e^{r t}+(k / r)\\left(e^{r t}-1\\right)\n$$ we have\r\n$$\r\nS(40)=(25,000)\\left(e^{3.2}-1\\right)=\\$ 588,313\r\n$$"
}
````

### Transformed record (`data/Platypus/scibench.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "For instance, suppose that one opens an individual retirement account (IRA) at age 25 and makes annual investments of $\\$ 2000$ thereafter in a continuous manner. Assuming a rate of return of $8 \\%$, what will be the balance in the IRA at age 65 ?",
  "response": "588313"
}
````
