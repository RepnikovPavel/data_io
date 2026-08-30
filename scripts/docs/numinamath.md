# numinamath

↑ [index](README.md) · ← [no_robots](no_robots.md) · [next → omnimath](omnimath.md)

**Script:** `pipe/clean_numinamath.py`

## Purpose

NuminaMath 1.5: ~900k competition-level math problems with CoT solutions, sourced from Chinese high-school exercises to international olympiads. The transform drops synthetic rows, invalid problems/solutions, and URL/translation artifacts; kept rows yield the full solution plus, for non-proofs, the short answer.

Created: 2025-02 · Domain: competition math _(date source: NuminaMath-1.5 release (repo createdAt 2025-02))_

## Before (raw storage)

- Source: HF `AI-MO/NuminaMath-1.5` (split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `problem` | UTF-8 text (arrow `string`) | the problem |
| `solution` | UTF-8 text (arrow `string`) | worked CoT solution |
| `answer` | UTF-8 text (arrow `string`) | short final answer ('proof' for proofs) |
| `synthetic` | boolean (arrow `bool`) | filter: synthetic rows are dropped |
| `problem_is_valid` | UTF-8 text (arrow `string`) | filter: must be 'Yes' |
| `solution_is_valid` | UTF-8 text (arrow `string`) | filter: must be 'Yes' |
| `question_type` | UTF-8 text (arrow `string`) | filter: 'proof' rows get no direct record |

## After (transformed)

- Location: `data/numinamath.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 811,921

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `noisy,cot` | 442,127 | full solution, source quality unverified | bulk competition-math reasoning signal |
| `noisy,direct` | 369,794 | short answer only, source quality unverified | answer-only variant of the same problems |

_exact counts (full scan of the jsonl file)_

## Keyword: `question_type` (input)

Problem form per the dataset card; drives whether a direct record is emitted.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `math-word-problem` | 21,453 | problem with a concrete answer | gets both a cot and a direct record |
| `proof` | 8,340 | prove/show statement | cot record only — no short answer exists |
| `MCQ` | 166 | multiple choice | rare here; answer is the option |
| `NaN` | 22 | — | — |
| `other` | 19 | uncategorized | treated like math-word-problem |

_estimated from the first 30,000 of 896,215 rows_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `AI-MO/NuminaMath-1.5`, split `train`

One row of HF `AI-MO/NuminaMath-1.5`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| problem | solution | answer | synthetic | problem_is_valid | solution_is_valid | question_type |
|---|---|---|---|---|---|---|
| ⏎Problem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number⏎⏎$$⏎x^{p}+y^{p}+z^{p}-x-y-z⏎$$⏎⏎is a product of exactly three distinct prime numbers.⏎ | ⏎Solution. Let $A=x^{p}+y^{p}+z^{p}-x-y-z$. For $p=2$, we take $x=y=4$ and $z=3$. Then $A=30=2 \cdot 3 \cdot 5$. For $p=3$ we can take $x=3$ and $y=2$ and $z=1$. Then again $A=30=2 \cdot 3 \cdot 5$. For $p=5$ we can take $x=2$ and $y=1$ and $z=1$. Again $A=30=2 \cdot 3 \cdot 5$.⏎⏎Assume now that $p \geqslant 7$. Working modulo 2 and modulo 3 we see that $A$ is divisible by both 2 and 3. Moreover, by Fermat's Little Theorem, we have⏎⏎$$⏎x^{p}+y^{p}+z^{p}-x-y-z \equiv x+y+z-x-y-z=0 \bmod p \text {. }⏎$$⏎⏎Therefore, by the given condition, we have to solve the equation⏎⏎$$⏎x^{p}+y^{p}+z^{p}-x-y-z… [truncated, 1552 chars total] | proof | False | Yes | Yes | math-word-problem |

### Raw — `question_type` = `math-word-problem`

One raw row with this value (same storage as above), shown as a table:

| problem | answer | question_type |
|---|---|---|
| ⏎Problem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number⏎⏎$$⏎x^{p}+y^{p}+z^{p}-x-y-z⏎$$⏎⏎is a product of exactly three distinct prime numbers.⏎ | proof | math-word-problem |

### Raw — `question_type` = `proof`

One raw row with this value (same storage as above), shown as a table:

| problem | answer | question_type |
|---|---|---|
| ⏎Problem 2. Let $a, b$ be two distinct real numbers and let $c$ be a positive real number such that⏎⏎$$⏎a^{4}-2019 a=b^{4}-2019 b=c .⏎$$⏎⏎Prove that $-\sqrt{c}<a b<0$.⏎ | proof | proof |

### Raw — `question_type` = `MCQ`

One raw row with this value (same storage as above), shown as a table:

| problem | answer | question_type |
|---|---|---|
| 3. Let $p_{1}, p_{2}, \ldots, p_{m}$ be the prime divisors of $a$. Among them, by condition, there are 3 and 5, for example, $\quad p_{1}=3, p_{2}=5$. Then $a=3^{s_{1}} \cdot 5^{s_{2}} \cdot p_{3}^{s_{3}} \cdot \ldots \cdot p_{m}^{s_{m}}$ and the total number of its divisors is $\left(s_{1}+1\right)\left(s_{2}+1\right) \cdot \ldots \cdot\left(s_{m}+1\right)=63=7 \cdot 9, s_{1} \geq 1, s_{2} \geq 1, s_{i} \geq 0, i=3,4, \ldots, m$. Then the following values of pairs $\left(s_{1} ; s_{2}\right)$ are possible:⏎⏎A) $s_{1}=2, s_{2}=2$ B) $s_{1}=2, s_{2}=6$, C) $s_{1}=6, s_{2}=2$, D) $s_{1}=2, s_{2}… [truncated, 1895 chars total] | 2^{6}\cdot3^{2}\cdot5^{2} | MCQ |

### Raw — `question_type` = `NaN`

One raw row with this value (same storage as above), shown as a table:

| problem | answer | question_type |
|---|---|---|
| 60. а) Имеется сеть дорог (черт. 5). Из точкл $A$ выходят $2^{1000}$ человек. Половина идет по направлению $l$, половина-⏎⏎![](https://cdn.mathpix.com/cropped/2024_05_21_68334347b64402497ba5g-029.jpg?height=232&width=323&top_left_y=1082&top_left_x=-1)⏎⏎Черт. 5. по направленню $m$. Дойдя до первого перекрестка, каждая группа разделяется: половина идет по наиравлениюо $l$, половина - по награвлению $m$. Такое же разделенне пронсходит на каждом перекрестке. Сколько людей прицет в три крайнпх слева перекрестка $B_{1}, B_{2}$ и $B_{3}$ тысячного ряда перекрестков?⏎⏎б) Решить эту задачу для всех пер… [truncated, 16938 chars total] |  | NaN |

### Raw — `question_type` = `other`

One raw row with this value (same storage as above), shown as a table:

| problem | answer | question_type |
|---|---|---|
| 5. (informatics) In a square matrix of size $N \times N$, all cells are filled with numbers from 1 to 5. A connected component in the matrix is defined as a set of cells that are filled with the same number, and between any two cells in the set, a path can be constructed. Cells can only be connected in a path if they are adjacent horizontally or vertically.⏎⏎In other words, a connected component of the matrix is a connected shape, where the cells are filled with the same number.⏎⏎For a given matrix, the task is to find the connected component consisting of the largest number of cells.⏎⏎\| 1 \| 2… [truncated, 3951 chars total] | 12 | other |

### Transformed — condition=`noisy,cot`

One line of `data/numinamath.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "noisy,cot", "instruction": "Problem 1. Find all prime numbers $p$ for which there exist positive integers $x, y$ and $z$ such that the number\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z\n$$\n\nis a product of exactly three distinct prime numbers.", "response": "Solution. Let $A=x^{p}+y^{p}+z^{p}-x-y-z$. For $p=2$, we take $x=y=4$ and $z=3$. Then $A=30=2 \\cdot 3 \\cdot 5$. For $p=3$ we can take $x=3$ and $y=2$ and $z=1$. Then again $A=30=2 \\cdot 3 \\cdot 5$. For $p=5$ we can take $x=2$ and $y=1$ and $z=1$. Again $A=30=2 \\cdot 3 \\cdot 5$.\n\nAssume now that $p \\geqslant 7$. Working modulo 2 and modulo 3 we see that $A$ is divisible by both 2 and 3. Moreover, by Fermat's Little Theorem, we have\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z \\equiv x+y+z-x-y-z=0 \\bmod p \\text {. }\n$$\n\nTherefore, by the given condition, we have to solve the equation\n\n$$\nx^{p}+y^{p}+z^{p}-x-y-z=… [truncated, 1550 chars total]"}
````

### Transformed — condition=`noisy,direct`

One line of `data/numinamath.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "noisy,direct", "instruction": "Problem 3. Find all the triples of integers $(a, b, c)$ such that the number\n\n$$\nN=\\frac{(a-b)(b-c)(c-a)}{2}+2\n$$\n\nis a power of 2016 .\n\n(A power of 2016 is an integer of the form $2016^{n}$, where $n$ is a non-negative integer.)", "response": "(,b,)=(k+2,k+1,k),k\\in\\mathbb{Z}"}
````

---

↑ [index](README.md) · ← [no_robots](no_robots.md) · [next → omnimath](omnimath.md)
