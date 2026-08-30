# flan

**Script:** `pipe_clustered/clean_flan.py`

## Purpose

FLAN v2 instruction-tuning collection (Open-Orca parquet dump). 14 subsets are included; every output parquet holds one (subset, task) pair. Few-shot/zero-shot option subsets are tagged direct, the two cot_* subsets cot.

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/Open-Orca/FLAN/<subset>/*.parquet`
- Format: parquet files per subset

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `_task_name` | string | source task id; becomes part of the output filename |
| `inputs` | string | prompt text -> instruction |
| `targets` | string | target text -> response |

## After (transformed)

- Location: `data_clustered/flan/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 4208 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 377,759,274

`condition` values used here:

- `direct` — 12 subsets: dialog/flan/niv2/t0 fsopt+fsnoopt+zsopt+zsnoopt
- `cot` — 2 subsets: cot_fsopt_data, cot_zsopt_data

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (`/mnt/hdd2/datasets_text/Open-Orca/FLAN/cot_fsopt_data/part.0.parquet`)

_first row of the subset's first parquet file; the transformed record below is row 1 of the parquet for that exact (subset, task) pair_

````text
{
  "_task_name": "cot_esnli_ii",
  "inputs": "The man be showing his toys to adults and not just kids.. So what could be the question?\nQuestion followed by answer: If \"A man wearing dark green shirt and sweatpants is showing off a stuffed toy.\" does that mean that \"A man is showing his stuffed toy to all the kids.\"?\nOptions:\n- yes\n- it is not possible to tell\n- no\nit is not possible to tell\n\nThe infant cannot be crying and asleep at the same time.\nThe question and answer are below.\nCan we conclude from \"Two young boys hold a crying infant while both are sitting in the same chair.\" that \"The infant is fast asleep.\"?\nOptions:\n- yes\n- no\n- i\n… [truncated, 1743 chars total]",
  "targets": "Premise: \"Woman skates in possession of puck.\"\nBased on this premise, can we conclude that the hypothesis \"The woman is figure skating.\" is true?\nOptions:\n- yes\n- it is not possible to tell\n- no\nno"
}
````

### Transformed record (`data_clustered/flan/cot_fsopt_data__cot_esnli_ii.parquet`, record 1)

````text
{
  "instruction": "The man be showing his toys to adults and not just kids.. So what could be the question?\nQuestion followed by answer: If \"A man wearing dark green shirt and sweatpants is showing off a stuffed toy.\" does that mean that \"A man is showing his stuffed toy to all the kids.\"?\nOptions:\n- yes\n- it is not possible to tell\n- no\nit is not possible to tell\n\nThe infant cannot be crying and asleep at the same time.\nThe question and answer are below.\nCan we conclude from \"Two young boys hold a crying infant while both are sitting in the same chair.\" that \"The infant is fast asleep.\"?\nOptions:\n- yes\n- no\n- i\n… [truncated, 1743 chars total]",
  "response": "Premise: \"Woman skates in possession of puck.\"\nBased on this premise, can we conclude that the hypothesis \"The woman is figure skating.\" is true?\nOptions:\n- yes\n- it is not possible to tell\n- no\nno",
  "condition": "cot"
}
````

### Transformed record (`data_clustered/flan/cot_fsopt_data__cot_esnli_ii.parquet`, record 2)

````text
{
  "instruction": "A man in a brown jacket and a white shirt looks to his side does not mean that he looks to his side to jump.\nThe question and answer are below.\nPremise: \"A man in a brown jacket and a white shirt looks to his side.\"\nHypothesis: \"A man in a brown jacket and a white shirt looks to his side to jump.\"\nDo we know that the hypothesis entailed by the premise?\nit is not possible to tell\n\nThe surfer might not be heading home and might just be beginning his journey.\nThe question and answer are below.\nGiven the sentence \"Male surfer with yellow surfboard and dog walking down picturesque beach and large r\n… [truncated, 1839 chars total]",
  "response": "Premise: \"Two construction workers working on a rooftop.\"\nBased on this premise, can we conclude that the hypothesis \"Men working on a roof.\" is true?\nOptions:\n- yes\n- it is not possible to tell\n- no\nyes",
  "condition": "cot"
}
````
