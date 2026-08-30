# ampsmathematica

**Script:** `pipe_clustered/clean_ampsmathematica.py`

## Purpose

AMPS Mathematica synthetic math exercises, read directly from the tar archive. Files under a `*_w_steps` task folder carry step-by-step answers (cot), the rest final answers (direct). Output is grouped one parquet per topic_subtask.

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/amps.tar.gz` (members `amps/mathematica/<topic>/<task>/*.txt`)
- Format: gzipped tar of small .txt files: 'Problem: ... Answer: ...'

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `raw_content` | string | whole file: 'Problem:' prefix stripped, split on the first 'Answer:' |

## After (transformed)

- Location: `data_clustered/ampsmathematica/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 137 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 4,830,500

`condition` values used here:

- `noisy,cot` — task folder ends with `_w_steps`: worked steps
- `noisy,direct` — other task folders: final answer only

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/amps.tar.gz` member `amps/mathematica/algebra/factor_polynomials_w_steps/3644.txt`)

````text
{
  "raw_content": "Problem:\nFactor the following quadratic: $-x^3-\\frac{x^2}{2}+141 x+648$\nAnswer:\n$\\begin{array}{l}\n \n\\begin{array}{l}\n \\text{Factor the following}: \\\\\n -x^3-\\frac{x^2}{2}+141 x+648 \\\\\n\\end{array}\n \\\\\n\\hline\n \n\\begin{array}{l}\n \\text{Put }\\text{each }\\text{term }\\text{in }-x^3-\\frac{x^2}{2}+141 x+648 \\text{over }\\text{the }\\text{common }\\text{denominator }2: -x^3-\\frac{x^2}{2}+141 x+648 \\text{= }-\\frac{2 x^3}{2}-\\frac{x^2}{2}+\\frac{282 x}{2}+\\frac{1296}{2}: \\\\\n -\\frac{2 x^3}{2}-\\frac{x^2}{2}+\\frac{282 x}{2}+\\frac{1296}{2} \\\\\n\\end{array}\n \\\\\n \n\\begin{array}{l}\n -\\frac{2 x^3}{2}-\\frac{x^2}{2}+\\fra\n… [truncated, 1637 chars total]"
}
````

### Transformed record (`data_clustered/ampsmathematica/algebra_arithmetic_series.parquet`, record 1)

````text
{
  "condition": "noisy,direct",
  "instruction": "Consider the arithmetic sequence defined by $a_1=\\frac{1}{12}$, and $a_n=a_{n-1}+10$ for $n > 1$. Compute the nth partial sum, $S_n=\\sum_{k=1}^n a_k$, where $n=19$.",
  "response": "$\\frac{20539}{12}$"
}
````

### Transformed record (`data_clustered/ampsmathematica/algebra_arithmetic_series.parquet`, record 2)

````text
{
  "condition": "noisy,direct",
  "instruction": "Consider the arithmetic sequence defined by $a_1=-\\frac{5}{8}$, and $a_n=a_{n-1}+-\\sqrt{3}$ for $n > 1$. Compute the nth partial sum, $S_n=\\sum_{k=1}^n a_k$, where $n=21$.",
  "response": "$\\frac{21}{2} \\left(-\\frac{5}{4}-20 \\sqrt{3}\\right)$"
}
````
