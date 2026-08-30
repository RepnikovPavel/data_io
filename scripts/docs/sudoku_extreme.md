# sudoku_extreme

**Script:** `pipe_clustered/clean_sudoku.py`

## Purpose

sudoku-extreme: millions of 81-cell Sudoku puzzles. The puzzle string ('.' -> '0') gets a fixed prompt prefix; the response is the solved grid. Teaches long-horizon constraint reasoning.

## Before (raw storage)

- Source: HF `sapientinc/sudoku-extreme` (file `train.csv`)
- Format: CSV: source,question,answer (81-char strings)

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question` | string | puzzle, 81 chars, '.' = empty cell |
| `answer` | string | solved grid, 81 chars |

## After (transformed)

- Location: `data_clustered/sudoku_extreme/all.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 1 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 3,831,994

`condition` values used here:

- `direct` — solved grid, no reasoning

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `sapientinc/sudoku-extreme` file `train.csv`)

````text
{
  "source": "puzzles4_forum_hardest_1905",
  "question": "5...27..9..41......1..5.3...92.6.8...5......66..7..29.8...7...2.......8...9..36..",
  "answer": "583427169974136528216859374792364851351298746648715293865971432137642985429583617",
  "rating": "18"
}
````

### Transformed record (`data_clustered/sudoku_extreme/all.parquet`, record 1)

````text
{
  "instruction": "Solve the Sudoku\n\n500027009004100000010050300092060800050000006600700290800070002000000080009003600",
  "response": "583427169974136528216859374792364851351298746648715293865971432137642985429583617",
  "condition": "direct"
}
````

### Transformed record (`data_clustered/sudoku_extreme/all.parquet`, record 2)

````text
{
  "instruction": "Solve the Sudoku\n\n020100904000020600003094005004970050010002407000040060006000000000000080090010706",
  "response": "827165934459328671163794825384976152615832497972541368236487519741659283598213746",
  "condition": "direct"
}
````
