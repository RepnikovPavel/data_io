# omnimath

**Script:** `pipe/clean_omnimath.py`

## Purpose

Omni-MATH olympiad-level problems (the test split is reused as training data). Each problem yields both the full solution (cot) and the short final answer (direct).

## Before (raw storage)

- Source: HF `KbsdJames/Omni-MATH` (split `test`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `problem` | string | the problem |
| `solution` | string | worked solution |
| `answer` | string | short final answer |

## After (transformed)

- Location: `data/omnimath.jsonl` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: JSONL — one JSON object per line, keys `condition`, `instruction`, `response` (all string)
- Rows: 8,856

`condition` values used here:

- `cot` — full worked solution
- `direct` — short final answer

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `KbsdJames/Omni-MATH`, split `test`)

````text
{
  "problem": "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\\le i ,j\\le n)$ satisfying:\n(1)For every $1\\le i ,j\\le n, x_{ij}=max\\{x_{i1},x_{i2},...,x_{ij}\\} $ or $ x_{ij}=max\\{x_{1j},x_{2j},...,x_{ij}\\}.$\n(2)For every $1\\le i \\le n$, there are at most $m$ indices $k$ with $x_{ik}=max\\{x_{i1},x_{i2},...,x_{ik}\\}.$\n(3)For every $1\\le j \\le n$, there are at most $m$ indices $k$ with $x_{kj}=max\\{x_{1j},x_{2j},...,x_{kj}\\}.$",
  "solution": "\nLet \\( n (\\geq 2) \\) be a positive integer. We aim to find the minimum \\( m \\) such that there exists \\( x_{ij} \\) (for \\( 1 \\leq i, j \\leq n \\)) satisfying the following conditions:\n1. For every \\( 1 \\leq i, j \\leq n \\), \\( x_{ij} = \\max \\{ x_{i1}, x_{i2}, \\ldots, x_{ij} \\} \\) or \\( x_{ij} = \\max \\{ x_{1j}, x_{2j}, \\ldots, x_{ij} \\} \\).\n2. For every \\( 1 \\leq i \\leq n \\), there are at most \\( m \\) indices \\( k \\) such that \\( x_{ik} = \\max \\{ x_{i1}, x_{i2}, \\ldots, x_{ik} \\} \\).\n3. For every \\( 1 \\leq j \\leq n \\), there are at most \\( m \\) indices \\( k \\) such that \\( x_{kj} = \\max \\{ x_{1j\n… [truncated, 1138 chars total]",
  "answer": "1 + \\left\\lceil \\frac{n}{2} \\right\\rceil"
}
````

### Transformed record (`data/omnimath.jsonl`, record 1)

````text
{
  "condition": "cot",
  "instruction": "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\\le i ,j\\le n)$ satisfying:\n(1)For every $1\\le i ,j\\le n, x_{ij}=max\\{x_{i1},x_{i2},...,x_{ij}\\} $ or $ x_{ij}=max\\{x_{1j},x_{2j},...,x_{ij}\\}.$\n(2)For every $1\\le i \\le n$, there are at most $m$ indices $k$ with $x_{ik}=max\\{x_{i1},x_{i2},...,x_{ik}\\}.$\n(3)For every $1\\le j \\le n$, there are at most $m$ indices $k$ with $x_{kj}=max\\{x_{1j},x_{2j},...,x_{kj}\\}.$",
  "response": "Let \\( n (\\geq 2) \\) be a positive integer. We aim to find the minimum \\( m \\) such that there exists \\( x_{ij} \\) (for \\( 1 \\leq i, j \\leq n \\)) satisfying the following conditions:\n1. For every \\( 1 \\leq i, j \\leq n \\), \\( x_{ij} = \\max \\{ x_{i1}, x_{i2}, \\ldots, x_{ij} \\} \\) or \\( x_{ij} = \\max \\{ x_{1j}, x_{2j}, \\ldots, x_{ij} \\} \\).\n2. For every \\( 1 \\leq i \\leq n \\), there are at most \\( m \\) indices \\( k \\) such that \\( x_{ik} = \\max \\{ x_{i1}, x_{i2}, \\ldots, x_{ik} \\} \\).\n3. For every \\( 1 \\leq j \\leq n \\), there are at most \\( m \\) indices \\( k \\) such that \\( x_{kj} = \\max \\{ x_{1j}\n… [truncated, 1137 chars total]"
}
````

### Transformed record (`data/omnimath.jsonl`, record 2)

````text
{
  "condition": "direct",
  "instruction": "Let $ n(\\ge2) $ be a positive integer. Find the minimum $ m $, so that there exists $x_{ij}(1\\le i ,j\\le n)$ satisfying:\n(1)For every $1\\le i ,j\\le n, x_{ij}=max\\{x_{i1},x_{i2},...,x_{ij}\\} $ or $ x_{ij}=max\\{x_{1j},x_{2j},...,x_{ij}\\}.$\n(2)For every $1\\le i \\le n$, there are at most $m$ indices $k$ with $x_{ik}=max\\{x_{i1},x_{i2},...,x_{ik}\\}.$\n(3)For every $1\\le j \\le n$, there are at most $m$ indices $k$ with $x_{kj}=max\\{x_{1j},x_{2j},...,x_{kj}\\}.$",
  "response": "1 + \\left\\lceil \\frac{n}{2} \\right\\rceil"
}
````
