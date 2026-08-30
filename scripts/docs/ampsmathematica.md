# ampsmathematica

↑ [index](README.md) · ← [acereason](acereason.md) · [next → openmathinstruct2](openmathinstruct2.md)

**Script:** `pipe_clustered/clean_ampsmathematica.py`

## Purpose

AMPS Mathematica: machine-generated math exercises with Mathematica-produced answers, read straight from the tar archive. Files under a `*_w_steps` task folder carry step-by-step answers (cot), the rest final answers (direct); output is grouped one parquet per topic_subtask.

Created: 2021-03 · Domain: synthetic math exercises (Mathematica) _(date source: AMPS, released with the MATH paper, arXiv:2103.03874)_

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/amps.tar.gz` (members `amps/mathematica/<topic>/<task>/*.txt`)
- Storage: gzip tar of small UTF-8 text files: 'Problem: ... Answer: ...'

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `raw_content` | UTF-8 text file inside a gzip tar | whole file: 'Problem:' prefix stripped, split on the first 'Answer:' |

## After (transformed)

- Location: `data_clustered/ampsmathematica/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 137 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 4,830,500

| column | type | meaning |
|---|---|---|
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | comma-separated tags (see keyword table below) |
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the target |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `noisy,direct` | 4,605,500 | other task folders: final answer only | answer-only math drill |
| `noisy,cot` | 225,000 | task folder ends with `_w_steps`: worked steps | step-wise math signal of mixed quality |

_exact counts (condition is constant per file; rows summed from parquet metadata)_

## Keyword: `task folder suffix` (input)

The `_w_steps` suffix on the tar member's task folder decides the condition (and the output file).

| value | rows | meaning | why it matters |
|---|---|---|---|
| `plain` | 4,605,500 | answer is the final result | tagged noisy,direct |
| `w_steps` | 225,000 | answer contains worked steps | tagged noisy,cot |

_exact (rows summed per file from metadata/line counts)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/amps.tar.gz` member `amps/mathematica/algebra/factor_polynomials_w_steps/3644.txt`

Content of `/mnt/hdd2/datasets_text/amps.tar.gz` member `amps/mathematica/algebra/factor_polynomials_w_steps/3644.txt` (gzip tar of small UTF-8 text files, 'Problem: ... Answer: ...'):

````text
Problem:
Factor the following quadratic: $-x^3-\frac{x^2}{2}+141 x+648$
Answer:
$\begin{array}{l}
 
\begin{array}{l}
 \text{Factor the following}: \\
 -x^3-\frac{x^2}{2}+141 x+648 \\
\end{array}
 \\
\hline
 
\begin{array}{l}
 \text{Put }\text{each }\text{term }\text{in }-x^3-\frac{x^2}{2}+141 x+648 \text{over }\text{the }\text{common }\text{denominator }2: -x^3-\frac{x^2}{2}+141 x+648 \text{= }-\frac{2 x^3}{2}-\frac{x^2}{2}+\frac{282 x}{2}+\frac{1296}{2}: \\
 -\frac{2 x^3}{2}-\frac{x^2}{2}+\frac{282 x}{2}+\frac{1296}{2} \\
\end{array}
 \\
 
\begin{array}{l}
 -\frac{2 x^3}{2}-\frac{x^2}{2}+\frac{282 x}{2}+\frac{1296}{2}=\frac{-2 x^3-x^2+282 x+1296}{2}: \\
 \frac{-2 x^3-x^2+282 x+1296}{2} \\
\end{array}
 \\
 
\begin{array}{l}
 \text{Factor }-1 \text{out }\text{of }-2 x^3-x^2+282 x+1296: \\
 \frac{\fbox{$-\left(2 x^3+x^2-282 x-1296\right)$}}{2} \\
\end{array}
 \\
 
\begin{array}{l}
 \text{The }\text{possible }\text{rational }\text{roots }\text{of }2 x^3+x^2-282 x-1296 \text{are }x=\pm \frac{1}{2},x=\pm \frac{3}{2},x=\pm \frac{9}{2},x=\pm \frac{27}{2},x=\pm \frac{81}{2},x=\pm 1,x=\pm 2,x=\pm 3,x=\pm 4,x=\pm 6,x=\pm 8,x=\pm 9,x=\pm 12,x=\pm 16,x=\pm 18,x=\pm 24,x=\pm 27,x=\pm 36,x=\pm 48,x=\pm 54,x=\pm 72,x=\pm 81,x=\pm 108,x=\pm 144,x=\pm 162,x=\pm 216,x=\pm 324,x=\pm 432,x=\pm 648,x=\pm 1296. \text{Of }\text{these, }x=\frac{27}{2},x=-6 \text{and }x=-8 \text{are }\text{roots. }\text{This }\text{gives }2 x-27,x+6 \text{and }x+8 \text{as }\text{all }\text{factors}: \\
 \fbox{$
\begin{array}{ll}
 \text{Answer:} &  \\
 \text{} & \frac{-\fbox{$(2 x-27) (x+6) (x+8)$}}{2} \\
\end{array}
$} \\
\end{array}
 \\
\end{array}$
````

### Transformed — condition=`noisy,direct`

One row of `data_clustered/ampsmathematica/algebra_arithmetic_series.parquet` (parquet table, columns: `condition`, `instruction`, `response`), shown as a table:

| condition | instruction | response |
|---|---|---|
| noisy,direct | Consider the arithmetic sequence defined by $a_1=\frac{1}{12}$, and $a_n=a_{n-1}+10$ for $n > 1$. Compute the nth partial sum, $S_n=\sum_{k=1}^n a_k$, where $n=19$. | $\frac{20539}{12}$ |

### Transformed — condition=`noisy,cot`

One row of `data_clustered/ampsmathematica/algebra_complete_square_w_steps.parquet` (parquet table, columns: `condition`, `instruction`, `response`), shown as a table:

| condition | instruction | response |
|---|---|---|
| noisy,cot | Given the equation $-9 x^2-5 x-6 y^2-6 y-3=0$, complete the square. | \begin{array}{l}⏎ ⏎\begin{array}{l}⏎ \text{Complete the square}: \\⏎ -6 y^2-6 y-9 x^2-5 x-3=0 \\⏎\end{array}⏎ \\⏎\hline⏎ ⏎\begin{array}{l}⏎ \text{Add }3 \text{to }\text{both }\text{sides}: \\⏎ -6 y^2-6 y-9 x^2-5 x=3 \\⏎\end{array}⏎ \\⏎ ⏎\begin{array}{l}⏎ \text{Group }\text{terms }\text{with }x \text{and }y \text{separately, }\text{leaving }\text{placeholder }\text{constants}: \\⏎ \left(-9 x^2-5 x+\underline{\text{   }}\right)+\left(-6 y^2-6 y+\underline{\text{   }}\right)=\underline{\text{   }}+3 \\⏎\end{array}⏎ \\⏎ ⏎\begin{array}{l}⏎ \left(-9 x^2-5 x+\underline{\text{   }}\right)=-9 \left(x^2… [truncated, 3359 chars total] |

---

↑ [index](README.md) · ← [acereason](acereason.md) · [next → openmathinstruct2](openmathinstruct2.md)
