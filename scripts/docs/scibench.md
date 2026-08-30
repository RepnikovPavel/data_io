# scibench

↑ [index](README.md) · ← [reclor](reclor.md) · [next → scienceqa](scienceqa.md)

**Script:** `pipe/clean_platypus/clean_scibench.py`

## Purpose

SciBench: college-level scientific problems from textbook JSON files (physics, chemistry, math; `*_sol.json` files carry worked solutions). The transform emits the worked solution when present (cot) and always the final numeric answer (direct).

Created: 2023-07 · Domain: college-level science problems _(date source: SciBench paper, arXiv:2307.10635)_

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/Platypus/scibench/dataset/original/*.json`
- Storage: one JSON array of objects per textbook file, UTF-8

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `problem_text` | JSON string — UTF-8 text | the problem |
| `solution` | JSON string — UTF-8 text | worked solution (cot record emitted only when non-empty) |
| `answer_latex` | JSON string — UTF-8 text | final answer in LaTeX (falls back to answer_number) |
| `answer_number` | JSON string — UTF-8 text | numeric final answer; used when answer_latex is redundant |

## After (transformed)

- Location: `data/Platypus/scibench.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 867

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 692 | final answer (answer_latex, simplified when it duplicates answer_number) | answer-only variant of every problem |
| `cot` | 175 | worked solution (only when the source has one) | scientific problem solving with steps |

_exact counts (full scan of the jsonl file)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/Platypus/scibench/dataset/original/atkins_sol.json`

First element of the JSON array in `/mnt/hdd2/datasets_text/Platypus/scibench/dataset/original/atkins_sol.json` (UTF-8 JSON file):

````json
{
  "problem_text": "The change in molar internal energy when $\\mathrm{CaCO}_3(\\mathrm{~s})$ as calcite converts to another form, aragonite, is $+0.21 \\mathrm{~kJ} \\mathrm{~mol}^{-1}$. Calculate the difference between the molar enthalpy and internal energy changes when the pressure is 1.0 bar given that the densities of the polymorphs are $2.71 \\mathrm{~g} \\mathrm{~cm}^{-3}$ and $2.93 \\mathrm{~g} \\mathrm{~cm}^{-3}$, respectively.",
  "solution": "The change in enthalpy when the transition occurs is\r\n$$\r\n\\begin{aligned}\r\n\\Delta H_{\\mathrm{m}} & =H_{\\mathrm{m}}(\\text { aragonite })-H_{\\mathrm{m}}(\\text { calcite }) \\\\\r\n& =\\left\\{U_{\\mathrm{m}}(\\mathrm{a})+p V_{\\mathrm{m}}(\\mathrm{a})\\right\\}-\\left\\{U_{\\mathrm{m}}(\\mathrm{c})+p V_{\\mathrm{m}}(\\mathrm{c})\\right\\} \\\\\r\n& =\\Delta U_{\\mathrm{m}}+p\\left\\{V_{\\mathrm{m}}(\\mathrm{a})-V_{\\mathrm{m}}(\\mathrm{c})\\right\\}\r\n\\end{aligned}\r\n$$\r\nwhere a denotes aragonite and c calcite. It follows by substituting $V_{\\mathrm{m}}=M / \\rho$ that\r\n$$\r\n\\Delta H_{\\mathrm{m}}-\\Delta U_{\\mathrm{m}}=p M\\left(\\frac… [truncated, 1157 chars total]",
  "answer_latex": " -0.28",
  "answer_number": "-0.28"
}
````

### Transformed — condition=`cot`

One line of `data/Platypus/scibench.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "cot", "instruction": "For instance, suppose that one opens an individual retirement account (IRA) at age 25 and makes annual investments of $\\$ 2000$ thereafter in a continuous manner. Assuming a rate of return of $8 \\%$, what will be the balance in the IRA at age 65 ?", "response": "We have $S_0=0, r=0.08$, and $k=\\$ 2000$, and we wish to determine $S(40)$. From Eq. $$\nS(t)=S_0 e^{r t}+(k / r)\\left(e^{r t}-1\\right)\n$$ we have\r\n$$\r\nS(40)=(25,000)\\left(e^{3.2}-1\\right)=\\$ 588,313\r\n$$"}
````

### Transformed — condition=`direct`

One line of `data/Platypus/scibench.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "For instance, suppose that one opens an individual retirement account (IRA) at age 25 and makes annual investments of $\\$ 2000$ thereafter in a continuous manner. Assuming a rate of return of $8 \\%$, what will be the balance in the IRA at age 65 ?", "response": "588313"}
````

---

↑ [index](README.md) · ← [reclor](reclor.md) · [next → scienceqa](scienceqa.md)
