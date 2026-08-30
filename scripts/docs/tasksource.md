# tasksource

**Script:** `pipe_clustered/clean_tasksource.py`

## Purpose

tasksource-instruct-v0: ~200 curated NLP tasks (NLI, classification, tagging, MCQ). Rows outside the curated TASK_SET are dropped; one output parquet per task.

## Before (raw storage)

- Source: HF `tasksource/tasksource-instruct-v0` (split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `task` | string | task id; filter (TASK_SET) and output filename |
| `inputs` | string | prompt -> instruction |
| `targets` | string | target -> response (trailing '.' removed) |

## After (transformed)

- Location: `data_clustered/tasksource/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 182 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 2,363,550

`condition` values used here:

- `direct` — short task target

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `tasksource/tasksource-instruct-v0`, split `train`)

````text
{
  "task": "reclor",
  "inputs": "With no explanation, chose the best option from \"A\", \"B\", \"C\" or \"D\". In rheumatoid arthritis, the body' s immune system misfunctions by attacking healthy cells in the joints causing the release of a hormone that in turn causes pain and swelling. This hormone is normally activated only in reaction to injury or infection. A new arthritis medication will contain a protein that inhibits the functioning of the hormone that causes pain and swelling in the joints. The statements above, if true, most strongly support which one of the following conclusions?\n\nA: Joint diseases other than rheumatoid art\n… [truncated, 1118 chars total]",
  "targets": "C."
}
````

### Transformed record (`data_clustered/tasksource/CONDAQA.parquet`, record 1)

````text
{
  "instruction": "With no explanation, label text_A→text_B with either \"DON'T KNOW\", \"NO\" or \"YES\".\ntext_A: In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While he vetoed what was an unprecedented eleven City Council ordinances that year, most narrowly were involved with specific financial practices considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimou\n… [truncated, 798 chars total]",
  "response": "NO",
  "condition": "direct"
}
````

### Transformed record (`data_clustered/tasksource/CONDAQA.parquet`, record 2)

````text
{
  "instruction": "With no explanation, label text_A→text_B with either \"DON'T KNOW\", \"NO\" or \"YES\".\ntext_A: In his first year as mayor, Medill received very little legislative resistance from the Chicago City Council. While he vetoed what was an unprecedented eleven City Council ordinances that year, most narrowly were involved with specific financial practices considered wasteful and none of the vetoes were overridden. He used his new powers to appoint the members of the newly constituted Chicago Board of Education and the commissioners of its constituted public library. His appointments were approved unanimou\n… [truncated, 842 chars total]",
  "response": "YES",
  "condition": "direct"
}
````
