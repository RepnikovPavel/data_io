# sudoku_extreme

↑ [index](README.md) · ← [openthoughts2](openthoughts2.md) · [next → tasksource](tasksource.md)

**Script:** `pipe_clustered/clean_sudoku.py`

## Purpose

sudoku-extreme (Sapient): 3.8M training puzzles mixing easy sets with the hardest community-collected ones; exact-deduped, unique solutions. The puzzle string ('.' -> '0') gets a fixed 'Solve the Sudoku' prefix; the response is the solved 81-char grid. Teaches long-horizon constraint reasoning.

Created: 2024-10 · Domain: logic puzzles (sudoku) _(date source: HF release (repo createdAt 2024-10; Sapient))_

## Before (raw storage)

- Source: HF `sapientinc/sudoku-extreme` (file `train.csv`)
- Storage: CSV (UTF-8): header `source,question,answer,rating`; puzzle/answer are 81-char strings

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question` | CSV field — UTF-8 text | puzzle, 81 chars, '.' = empty cell ('.' -> '0' in the output) |
| `answer` | CSV field — UTF-8 text | solved grid, 81 chars -> response |

## After (transformed)

- Location: `data_clustered/sudoku_extreme/all.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 3,831,994

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the target |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 3,831,994 | solved grid, no reasoning | pure pattern/constraint output, no intermediate steps |

_exact counts (condition column read in full)_

## Keyword: `source` (input)

Puzzle collection the row came from (per the dataset card: puzzles0-2 are easy, puzzles3+ are the hardest known). Unused by the transform; shown for context.

| value | rows |
|---|---|
| `puzzles4_forum_hardest_1905` | 96,721 |
| `01_file1` | 47,837 |
| `puzzles1_unbiased` | 46,236 |
| `puzzles0_kaggle` | 4,523 |
| `puzzles2_17_clue` | 2,328 |
| `puzzles5_forum_hardest_1905_11+` | 2,268 |
| `puzzles3_magictour_top1465` | 64 |
| `puzzles6_forum_hardest_1106` | 23 |

_estimated from the first 200,000 data rows of the CSV_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `sapientinc/sudoku-extreme` file `train.csv`

Header + first data row of HF `sapientinc/sudoku-extreme` file `train.csv` (CSV, UTF-8):

````text
source,question,answer,rating
puzzles4_forum_hardest_1905,5...27..9..41......1..5.3...92.6.8...5......66..7..29.8...7...2.......8...9..36..,583427169974136528216859374792364851351298746648715293865971432137642985429583617,18
````

### Transformed — condition=`direct`

One row of `data_clustered/sudoku_extreme/all.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| Solve the Sudoku⏎⏎500027009004100000010050300092060800050000006600700290800070002000000080009003600 | 583427169974136528216859374792364851351298746648715293865971432137642985429583617 | direct |

---

↑ [index](README.md) · ← [openthoughts2](openthoughts2.md) · [next → tasksource](tasksource.md)
