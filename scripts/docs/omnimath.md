# omnimath

↑ [index](README.md) · ← [numinamath](numinamath.md) · [next → principia_collection](principia_collection.md)

**Script:** `pipe/clean_omnimath.py`

## Purpose

Omni-MATH: 4,428 olympiad-level problems spanning 33 sub-domains and 10 difficulty levels, published as a benchmark — here the test split is reused as training data. Each problem yields the full solution (cot) and the short final answer (direct).

Created: 2024-10 · Domain: olympiad math _(date source: Omni-MATH paper, arXiv:2410.07985)_

## Before (raw storage)

- Source: HF `KbsdJames/Omni-MATH` (split `test`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `problem` | UTF-8 text (arrow `string`) | the problem |
| `solution` | UTF-8 text (arrow `string`) | worked solution |
| `answer` | UTF-8 text (arrow `string`) | short final answer |

## After (transformed)

- Location: `data/omnimath.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: JSONL — one JSON object per line, UTF-8; keys `condition`, `instruction`, `response`
- Rows: 8,856

| column | type | meaning |
|---|---|---|
| `condition` | JSON string — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | JSON string — UTF-8 text | the prompt |
| `response` | JSON string — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `cot` | 4,428 | full worked solution | hardest tier of reasoning training data |
| `direct` | 4,428 | short final answer | answer-only variant of the same problems |

_exact counts (full scan of the jsonl file)_

## Keyword: `source` (input)

Originating competition per problem (unused by the transform; shown for context).

| value | rows |
|---|---|
| `HMMT_2` | 1,385 |
| `HMMT_11` | 896 |
| `pascal` | 249 |
| `fermat` | 232 |
| `cayley` | 201 |
| `imo_shortlist` | 190 |
| `usamo` | 133 |
| `china_team_selection_test` | 106 |
| `putnam` | 101 |
| `imo` | 74 |
| `imc` | 68 |
| `apmoapmo_sol` | 61 |

… 67 distinct values total, top 12 shown.

_exact counts (all 4,428 rows)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `KbsdJames/Omni-MATH`, split `test`

One row of HF `KbsdJames/Omni-MATH`, split `test` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| problem | solution | answer |
|---|---|---|
| Let $ n(\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\le i ,j\le n)$ satisfying:⏎(1)For every $1\le i ,j\le n, x_{ij}=max\{x_{i1},x_{i2},...,x_{ij}\} $ or $ x_{ij}=max\{x_{1j},x_{2j},...,x_{ij}\}.$⏎(2)For every $1\le i \le n$, there are at most $m$ indices $k$ with $x_{ik}=max\{x_{i1},x_{i2},...,x_{ik}\}.$⏎(3)For every $1\le j \le n$, there are at most $m$ indices $k$ with $x_{kj}=max\{x_{1j},x_{2j},...,x_{kj}\}.$ | ⏎Let \( n (\geq 2) \) be a positive integer. We aim to find the minimum \( m \) such that there exists \( x_{ij} \) (for \( 1 \leq i, j \leq n \)) satisfying the following conditions:⏎1. For every \( 1 \leq i, j \leq n \), \( x_{ij} = \max \{ x_{i1}, x_{i2}, \ldots, x_{ij} \} \) or \( x_{ij} = \max \{ x_{1j}, x_{2j}, \ldots, x_{ij} \} \).⏎2. For every \( 1 \leq i \leq n \), there are at most \( m \) indices \( k \) such that \( x_{ik} = \max \{ x_{i1}, x_{i2}, \ldots, x_{ik} \} \).⏎3. For every \( 1 \leq j \leq n \), there are at most \( m \) indices \( k \) such that \( x_{kj} = \max \{ x_{1j… [truncated, 1138 chars total] | 1 + \left\lceil \frac{n}{2} \right\rceil |

### Transformed — condition=`cot`

One line of `data/omnimath.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "cot", "instruction": "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\\le i ,j\\le n)$ satisfying:\n(1)For every $1\\le i ,j\\le n, x_{ij}=max\\{x_{i1},x_{i2},...,x_{ij}\\} $ or $ x_{ij}=max\\{x_{1j},x_{2j},...,x_{ij}\\}.$\n(2)For every $1\\le i \\le n$, there are at most $m$ indices $k$ with $x_{ik}=max\\{x_{i1},x_{i2},...,x_{ik}\\}.$\n(3)For every $1\\le j \\le n$, there are at most $m$ indices $k$ with $x_{kj}=max\\{x_{1j},x_{2j},...,x_{kj}\\}.$", "response": "Let \\( n (\\geq 2) \\) be a positive integer. We aim to find the minimum \\( m \\) such that there exists \\( x_{ij} \\) (for \\( 1 \\leq i, j \\leq n \\)) satisfying the following conditions:\n1. For every \\( 1 \\leq i, j \\leq n \\), \\( x_{ij} = \\max \\{ x_{i1}, x_{i2}, \\ldots, x_{ij} \\} \\) or \\( x_{ij} = \\max \\{ x_{1j}, x_{2j}, \\ldots, x_{ij} \\} \\).\n2. For every \\( 1 \\leq i \\leq n \\), there are at most \\( m \\) indices \\( k \\) such that \\( x_{ik} = \\max \\{ x_{i1}, x_{i2}, \\ldots, x_{ik} \\} \\).\n3. For every \\( 1 \\leq j \\leq n \\), there are at most \\( m \\) indices \\( k \\) such that \\( x_{kj} = \\max \\{ x_{1j}… [truncated, 1137 chars total]"}
````

### Transformed — condition=`direct`

One line of `data/omnimath.jsonl` (JSONL — one JSON object per line, UTF-8):

````jsonl
{"condition": "direct", "instruction": "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\\le i ,j\\le n)$ satisfying:\n(1)For every $1\\le i ,j\\le n, x_{ij}=max\\{x_{i1},x_{i2},...,x_{ij}\\} $ or $ x_{ij}=max\\{x_{1j},x_{2j},...,x_{ij}\\}.$\n(2)For every $1\\le i \\le n$, there are at most $m$ indices $k$ with $x_{ik}=max\\{x_{i1},x_{i2},...,x_{ik}\\}.$\n(3)For every $1\\le j \\le n$, there are at most $m$ indices $k$ with $x_{kj}=max\\{x_{1j},x_{2j},...,x_{kj}\\}.$", "response": "1 + \\left\\lceil \\frac{n}{2} \\right\\rceil"}
````

---

↑ [index](README.md) · ← [numinamath](numinamath.md) · [next → principia_collection](principia_collection.md)
