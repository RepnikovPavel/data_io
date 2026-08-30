# openmathinstruct2

**Script:** `pipe_clustered/clean_openmathinstruct2.py`

## Purpose

OpenMathInstruct-2 (nvidia): ~14M synthetic math solutions. Every row goes to cot.parquet (full generated solution); rows not derived from the original math/gsm8k data also go to direct.parquet (short expected answer).

## Before (raw storage)

- Source: HF `nvidia/OpenMathInstruct-2` (split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `problem` | string | the problem -> instruction |
| `generated_solution` | string | model-generated solution (cot.parquet response) |
| `expected_answer` | string | short answer (direct.parquet response) |
| `problem_source` | string | filter: rows from 'math'/'gsm8k' are excluded from direct.parquet |

## After (transformed)

- Location: `data_clustered/openmathinstruct2/cot.parquet`, `data_clustered/openmathinstruct2/direct.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 2 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 25,023,023

`condition` values used here:

- `synth,cot` — cot.parquet: full generated solution
- `synth,direct` — direct.parquet: short expected answer (non-original sources only)

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `nvidia/OpenMathInstruct-2`, split `train`)

````text
{
  "problem": "Ava is planning a camping trip with her friends. She wants to make sure they have enough granola bars for snacks. There will be five people total: Ava, her two friends, and her parents. They will spend 3 days and 2 nights at the campsite, and they plan to have 2 granola bars per person for breakfast and 1 granola bar per person for an afternoon snack each day. How many granola bars will Ava need to pack in total for the entire trip?",
  "generated_solution": "There will be a total of 5 people.\nEach person needs 2 granola bars for breakfast and 1 granola bar for snack. This amounts to a total of 3 granola bars per person per day.\nSince the trip is 3 days long, each person will need 3 granola bars/day * 3 days = 9 granola bars.\nSo for 5 people, Ava will need 5 * 9 = 45 granola bars.\nThus, Ava will need to pack \\boxed{45} granola bars in total for the entire trip.",
  "expected_answer": "45",
  "problem_source": "augmented_gsm8k"
}
````

### Transformed record (`data_clustered/openmathinstruct2/cot.parquet`, record 1)

````text
{
  "instruction": "Ava is planning a camping trip with her friends. She wants to make sure they have enough granola bars for snacks. There will be five people total: Ava, her two friends, and her parents. They will spend 3 days and 2 nights at the campsite, and they plan to have 2 granola bars per person for breakfast and 1 granola bar per person for an afternoon snack each day. How many granola bars will Ava need to pack in total for the entire trip?",
  "response": "There will be a total of 5 people.\nEach person needs 2 granola bars for breakfast and 1 granola bar for snack. This amounts to a total of 3 granola bars per person per day.\nSince the trip is 3 days long, each person will need 3 granola bars/day * 3 days = 9 granola bars.\nSo for 5 people, Ava will need 5 * 9 = 45 granola bars.\nThus, Ava will need to pack \\boxed{45} granola bars in total for the entire trip.",
  "condition": "synth,cot"
}
````

### Transformed record (`data_clustered/openmathinstruct2/cot.parquet`, record 2)

````text
{
  "instruction": "Find the sum of all three-digit positive integers whose cubes end with the digits 125.",
  "response": "To find the sum of all three-digit positive integers whose cubes end with the digits 125, let's analyze how a number's cube can end in 125.\n\nA number $n$ whose cube ends in 125 must be of the form $n = 5k$, where $k$ is an integer. This is because the last digit of $n^3$ is determined by the last digit of $n$, and only numbers ending in 5 have cubes that end in 5.\n\nNow, consider the last two digits of $n^3$, which are determined by the last two digits of $n$. The only possibilities for the last two digits of $n$ that result in a cube ending in 25 are 25, 75, and 25 + 50 = 75, 75 + 50 = 25 + 10\n… [truncated, 1181 chars total]",
  "condition": "synth,cot"
}
````
