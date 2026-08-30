# openthoughts2

↑ [index](README.md) · ← [openmathinstruct2](openmathinstruct2.md) · [next → sudoku_extreme](sudoku_extreme.md)

**Script:** `pipe_clustered/clean_openthoughts2.py`

## Purpose

OpenThoughts2-1M: 1.1M synthetic reasoning traces (math, science, code, puzzles) with R1-style `<think>` blocks. The transform drops code-related sources and code-looking rows, strips the think block, and keeps the remaining answer.

Created: 2025-04 · Domain: synthetic reasoning traces _(date source: OpenThoughts2 release (blog 'thinkagain', 2025-04))_

## Before (raw storage)

- Source: HF `open-thoughts/OpenThoughts2-1M` (split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `conversations` | arrow list — arrow struct {from: UTF-8 text (arrow `string`), value: UTF-8 text (arrow `string`)} | exactly one user + one assistant turn; both used |
| `source` | UTF-8 text (arrow `string`) | filter: code/math-duplicate sources dropped (dolphin, magicoder, sharegpt, nvidia_math, ...) |

## After (transformed)

- Location: `data_clustered/openthoughts2/all.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 892,168

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the target |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `synth,cot` | 892,168 | reasoning trace with <think> block removed | large synthetic reasoning corpus, code filtered out |

_exact counts (condition column read in full)_

## Keyword: `source` (input)

Upstream data source; the REMOVE_SOURCES set (dolphin, evolcodegolf, glaive, magicoder, sharegpt, codefeedback, nvidia_math) is dropped as code or duplicate.

| value | rows |
|---|---|
| `None` | 41,249 |
| `evolcodegolf` | 1,294 |
| `codefeedback` | 1,228 |
| `glaive` | 1,207 |
| `magicoder` | 1,195 |
| `sharegpt` | 1,119 |
| `automath` | 591 |
| `tiger_math` | 577 |
| `dolphin` | 523 |
| `nvidia_math` | 518 |
| `tiger_lab_math` | 499 |

_estimated from the first 50,000 of 1,143,205 rows_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `open-thoughts/OpenThoughts2-1M`, split `train`

One row of HF `open-thoughts/OpenThoughts2-1M`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| conversations | source |
|---|---|
| [{"from": "user", "value": "Given the matrix equation \n\n\\[\nC \\times \\begin{pmatrix} 9 & 1 \\\\ 4 & 6 \\\\ 3 & 4 \\end{pmatrix} = \\begin{pmatrix} 9 & 1 \\\\ 4 & 6 \\\\ 3 & 4 \\end{pmatrix},\n\\]\n\nwhere \\( C \\) is a \\( 3 \\times 3 \\) matrix, determine the conditions under which \\( C \\) can exist. Additionally, find the eigenvalues of \\( C \\) if \\( v_1 = \\begin{pmatrix} 9 \\\\ 4 \\\\ 3 \\end{pmatrix} \\) and \\( v_2 = \\begin{pmatrix} 1 \\\\ 6 \\\\ 4 \\end{pmatrix} \\) are eigenvectors of \\( C \\). What can you conclude about the uniqueness of \\( C \\)?"}, {"from": "assistant", "value": "<think>\nOkay, so I have this matrix equation here: C times a 3x2 matrix equals the same 3x2 matrix. And C is supposed to be a 3x3 matrix. The question is asking about the conditions under which C can exist. Then, they want me to find the eigenvalues of C if v1 and v2 are eigenvectors, and also comment on the uniqueness of C.\n\nFirst, let me parse the problem. The equation is C multiplied by a 3x2 matrix resulting in the same 3x2 matrix. But matrix multiplication usually requires that the number of columns of the first matrix matches the number of rows of the second. Since C is 3x3, multiplying … [truncated, 18775 chars total]"}] | automath |

### Raw — `source` = `None`

One raw row with this value (same storage as above), shown as a table:

| conversations | source |
|---|---|
| [{"from": "user", "value": "Return your final response within \\boxed{}. Find [the decimal form of] the largest prime divisor of $100111011_6$.\n"}, {"from": "assistant", "value": "<think>\nOkay, so I need to find the largest prime divisor of the number 100111011 in base 6, converted to decimal form. Hmm, let me start by understanding the steps involved here. First, I need to convert the base 6 number to base 10 (decimal), right? Once I have the decimal number, I can then find its prime divisors and determine which one is the largest. \n\nLet me start with the conversion process. The number given is 100111011_6. To convert this to decimal, I should expand each digit multiplied by 6 raised to the power of its position, starting from the rightmost digit which is position 0. L… [truncated, 9957 chars total]"}] | ∅ (null) |

### Raw — `source` = `evolcodegolf`

One raw row with this value (same storage as above), shown as a table:

| conversations | source |
|---|---|
| [{"from": "user", "value": "Determine if the boundaries of two rectangles intersect. Your program should return True if any part of one rectangle touches or overlaps the other, and False if they are completely separate. Rectangles that only share a corner are considered to intersect. Input consists of two pairs of coordinates representing the bottom-left and top-right corners of the rectangles, provided as eight floats or integers in any order. Output a boolean value indicating the intersection status, which can be expressed as True or False.\n\nTest cases are formatted as (x1, y1, x2, y2, x3, y3, x4, y4):\nThe following in… [truncated, 779 chars total]"}, {"from": "assistant", "value": "<think>\nOkay, I need to determine if two rectangles intersect, including touching edges or corners. The input is two pairs of coordinates for each rectangle. Let's think about how to approach this.\n\nFirst, rectangles are defined by their bottom-left and top-right corners. Wait, but the input could have the coordinates in any order, right? Oh, wait the problem statement says \"two pairs of coordinates representing the bottom-left and top-right corners\". So each rectangle is given as two points: (x1,y1) bottom-left and (x2,y2) top-right. Wait, but when the user inputs, they might not be ordered c… [truncated, 35930 chars total]"}] | evolcodegolf |

### Transformed — condition=`synth,cot`

One row of `data_clustered/openthoughts2/all.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| Return your final response within \boxed{}. Find [the decimal form of] the largest prime divisor of $100111011_6$.⏎ | ⏎⏎The decimal form of the number \(100111011_6\) is calculated by expanding its digits:⏎⏎\[⏎1 \cdot 6^8 + 0 \cdot 6^7 + 0 \cdot 6^6 + 1 \cdot 6^5 + 1 \cdot 6^4 + 1 \cdot 6^3 + 0 \cdot 6^2 + 1 \cdot 6^1 + 1 \cdot 6^0⏎\]⏎⏎Computing each term:⏎\[⏎1 \cdot 1679616 + 1 \cdot 7776 + 1 \cdot 1296 + 1 \cdot 216 + 1 \cdot 6 + 1 \cdot 1 = 1688911⏎\]⏎⏎Next, factorizing \(1688911\):⏎\[⏎1688911 = 7 \times 241273⏎\]⏎\[⏎241273 = 31 \times 7783⏎\]⏎\[⏎7783 = 43 \times 181⏎\]⏎⏎The prime factors are \(7\), \(31\), \(43\), and \(181\). The largest prime divisor is \(\boxed{181}\). | synth,cot |

---

↑ [index](README.md) · ← [openmathinstruct2](openmathinstruct2.md) · [next → sudoku_extreme](sudoku_extreme.md)
