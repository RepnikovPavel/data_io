# numinamath

**Script:** `pipe/clean_numinamath.py`

## Purpose

Large competition/olympiad math corpus (NuminaMath-1.5). Synthetic rows, invalid problems/solutions, and rows with URLs or translation artifacts are dropped. Each kept row yields the full solution plus, for non-proof rows, the short final answer.

## Before (raw storage)

- Source: HF `AI-MO/NuminaMath-1.5` (split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `problem` | string | the problem |
| `solution` | string | worked solution |
| `answer` | string | short final answer ('proof' for proofs) |
| `synthetic` | bool | filter: synthetic rows are dropped |
| `problem_is_valid / solution_is_valid` | string | filter: must both be 'Yes' |
| `question_type` | string | filter: 'proof' rows get no direct record |

## After (transformed)

- Location: `data/numinamath.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 811,921

`condition` values used here:

- `noisy,cot` — full solution, source quality unverified
- `noisy,direct` — short answer only, source quality unverified

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `AI-MO/NuminaMath-1.5`, split `train`)

````text
{
  "problem": "\nProblem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z\n$$\n\nis a product of exactly three distinct prime numbers.\n",
  "solution": "\nSolution. Let $A=x^{p}+y^{p}+z^{p}-x-y-z$. For $p=2$, we take $x=y=4$ and $z=3$. Then $A=30=2 \\cdot 3 \\cdot 5$. For $p=3$ we can take $x=3$ and $y=2$ and $z=1$. Then again $A=30=2 \\cdot 3 \\cdot 5$. For $p=5$ we can take $x=2$ and $y=1$ and $z=1$. Again $A=30=2 \\cdot 3 \\cdot 5$.\n\nAssume now that $p \\geqslant 7$. Working modulo 2 and modulo 3 we see that $A$ is divisible by both 2 and 3. Moreover, by Fermat's Little Theorem, we have\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z \\equiv x+y+z-x-y-z=0 \\bmod p \\text {. }\n$$\n\nTherefore, by the given condition, we have to solve the equation\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z\n… [truncated, 1552 chars total]",
  "answer": "proof",
  "synthetic": false,
  "problem_is_valid": "Yes",
  "solution_is_valid": "Yes",
  "question_type": "math-word-problem"
}
````

### Transformed record (`data/numinamath.jsonl`, record 1)

````text
{
  "condition": "noisy,cot",
  "instruction": "Problem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z\n$$\n\nis a product of exactly three distinct prime numbers.",
  "response": "Solution. Let $A=x^{p}+y^{p}+z^{p}-x-y-z$. For $p=2$, we take $x=y=4$ and $z=3$. Then $A=30=2 \\cdot 3 \\cdot 5$. For $p=3$ we can take $x=3$ and $y=2$ and $z=1$. Then again $A=30=2 \\cdot 3 \\cdot 5$. For $p=5$ we can take $x=2$ and $y=1$ and $z=1$. Again $A=30=2 \\cdot 3 \\cdot 5$.\n\nAssume now that $p \\geqslant 7$. Working modulo 2 and modulo 3 we see that $A$ is divisible by both 2 and 3. Moreover, by Fermat's Little Theorem, we have\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z \\equiv x+y+z-x-y-z=0 \\bmod p \\text {. }\n$$\n\nTherefore, by the given condition, we have to solve the equation\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z=\n… [truncated, 1550 chars total]"
}
````

### Transformed record (`data/numinamath.jsonl`, record 2)

````text
{
  "condition": "noisy,cot",
  "instruction": "Problem 2. Let $a, b$ and $c$ be positive real numbers. Prove that\n\n$$\n\\frac{8}{(a+b)^{2}+4 a b c}+\\frac{8}{(b+c)^{2}+4 a b c}+\\frac{8}{(c+a)^{2}+4 a b c}+a^{2}+b^{2}+c^{2} \\geq \\frac{8}{a+3}+\\frac{8}{b+3}+\\frac{8}{c+3}\n$$",
  "response": "Solution. Since $2 a b \\leq a^{2}+b^{2}$, it follows that $(a+b)^{2} \\leq 2\\left(a^{2}+b^{2}\\right)$\n\nand $4 a b c \\leq 2 c\\left(a^{2}+b^{2}\\right)$, for any positive reals $a, b, c$.\n\nAdding these inequalities, we find\n\n$$\n(a+b)^{2}+4 a b c \\leq 2\\left(a^{2}+b^{2}\\right)(c+1)\n$$\n\nso that\n\n$$\n\\frac{8}{(a+b)^{2}+4 a b c} \\geq \\frac{4}{\\left(a^{2}+b^{2}\\right)(c+1)}\n$$\n\nUsing the AM-GM inequality, we have\n\n$$\n\\frac{4}{\\left(a^{2}+b^{2}\\right)(c+1)}+\\frac{a^{2}+b^{2}}{2} \\geq 2 \\sqrt{\\frac{2}{c+1}}=\\frac{4}{\\sqrt{2(c+1)}}\n$$\n\nrespectively\n\n$$\n\\frac{c+3}{8}=\\frac{(c+1)+2}{8} \\geq \\frac{\\sqrt{2(c+1\n… [truncated, 884 chars total]"
}
````
