# math_train

↑ [index](README.md) · ← [gsm8k_train](gsm8k_train.md) · [next → natural_reasoning](natural_reasoning.md)

**Script:** `pipe/clean_math_train.py`

## Purpose

MATH (Hendrycks): 12.5k competition problems in 7 subjects with LaTeX worked solutions (train splits used here). Each problem yields the full solution (cot) plus the last `\boxed{...}` content as a bare answer (direct).

## Before (raw storage)

- Source: HF `EleutherAI/hendrycks_math` (7 subject configs, split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache, one config per subject

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `problem` | UTF-8 text (arrow `string`) | LaTeX problem statement |
| `solution` | UTF-8 text (arrow `string`) | worked solution containing `\boxed{...}` |

## After (transformed)

- Location: `data/math_train.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 14,996

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `cot` | 7,500 | full worked LaTeX solution | teaches competition-level step-by-step reasoning |
| `direct` | 7,496 | content of the last `\boxed{...}` (row omitted when no boxed answer) | same problems as answer-only supervision |

_exact counts (full scan of the jsonl file)_

## Keyword: `subject config` (input)

The 7 HF configs (subjects) concatenated by the script.

| value | rows |
|---|---|
| `algebra` | 1,744 |
| `intermediate_algebra` | 1,295 |
| `prealgebra` | 1,205 |
| `geometry` | 870 |
| `number_theory` | 869 |
| `counting_and_probability` | 771 |
| `precalculus` | 746 |

_exact (HF split sizes, train split per config)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `EleutherAI/hendrycks_math` (config `algebra`), split `train`

One row of HF `EleutherAI/hendrycks_math` (config `algebra`), split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| problem | solution |
|---|---|
| Let \[f(x) = \left\{⏎\begin{array}{cl} ax+3, &\text{ if }x>2, \\⏎x-5 &\text{ if } -2 \le x \le 2, \\⏎2x-b &\text{ if } x <-2.⏎\end{array}⏎\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper). | For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$. |

### Transformed — condition=`cot`

One line of `data/math_train.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "cot", "instruction": "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).", "response": "For the piecewise function to be continuous, the cases must \"meet\" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \\Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\\boxed{0}$."}
````

### Transformed — condition=`direct`

One line of `data/math_train.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2, \\\\\nx-5 &\\text{ if } -2 \\le x \\le 2, \\\\\n2x-b &\\text{ if } x <-2.\n\\end{array}\n\\right.\\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).", "response": "0"}
````

---

↑ [index](README.md) · ← [gsm8k_train](gsm8k_train.md) · [next → natural_reasoning](natural_reasoning.md)
