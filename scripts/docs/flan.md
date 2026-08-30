# flan

↑ [index](README.md) · ← [theoremqa](theoremqa.md) · [next → synth](synth.md)

**Script:** `pipe_clustered/clean_flan.py`

## Purpose

FLAN v2 instruction-tuning collection (Open-Orca parquet dump): templated tasks from the Flan/T0/NIV2/CoT/dialog submixtures, in few-shot/zero-shot and with/without-options variants. 14 subsets are included; each output parquet holds one (subset, task) pair.

Created: 2023-01 · Domain: instruction-tuning mixture _(date source: FLAN v2 collection, arXiv:2301.13688 (Open-Orca parquet dump, 2023-07))_

## Before (raw storage)

- Source: `/mnt/hdd2/datasets_text/Open-Orca/FLAN/<subset>/*.parquet`
- Storage: one parquet file set per subset directory

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `_task_name` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | source task id; becomes part of the output filename |
| `inputs` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | prompt text -> instruction |
| `targets` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | target text -> response |

## After (transformed)

- Location: `data_clustered/flan/*.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 4208 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 377,759,274

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `large_string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `large_string` — UTF-8 text | the target |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `large_string` — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `direct` | 377,480,800 | 12 subsets: dialog/flan/niv2/t0 in fsopt/fsnoopt/zsopt/zsnoopt variants | classic instruction-following supervision |
| `cot` | 278,474 | 2 subsets: cot_fsopt_data, cot_zsopt_data | chain-of-thought prompts with reasoning targets |

_exact counts (condition is constant per file; rows summed from parquet metadata)_

## Keyword: `subset` (input)

Subset directory = submixture x prompt style (fs = few-shot, zs = zero-shot, opt/noopt = answer options present/absent). One example pair below is drawn from cot_fsopt_data.

| value | rows |
|---|---|
| `t0_fsopt_data` | 65,903,143 |
| `flan_fsopt_data` | 64,227,718 |
| `t0_zsopt_data` | 41,652,381 |
| `t0_zsnoopt_data` | 39,662,627 |
| `flan_zsopt_data` | 38,970,972 |
| `flan_zsnoopt_data` | 37,384,598 |
| `flan_fsnoopt_data` | 34,258,827 |
| `t0_fsnoopt_data` | 32,191,051 |
| `niv2_fsopt_data` | 10,058,408 |
| `dialog_fsopt_data` | 5,425,015 |
| `niv2_zsopt_data` | 5,030,900 |
| `dialog_zsopt_data` | 2,715,160 |
| `cot_fsopt_data` | 182,904 |
| `cot_zsopt_data` | 95,570 |

_exact (rows summed per file from metadata/line counts)_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — `/mnt/hdd2/datasets_text/Open-Orca/FLAN/cot_fsopt_data/part.0.parquet`

One row of `/mnt/hdd2/datasets_text/Open-Orca/FLAN/cot_fsopt_data/part.0.parquet` (parquet table), shown with the columns the transform reads:

| _task_name | inputs | targets |
|---|---|---|
| cot_esnli_ii | The man be showing his toys to adults and not just kids.. So what could be the question?⏎Question followed by answer: If "A man wearing dark green shirt and sweatpants is showing off a stuffed toy." does that mean that "A man is showing his stuffed toy to all the kids."?⏎Options:⏎- yes⏎- it is not possible to tell⏎- no⏎it is not possible to tell⏎⏎The infant cannot be crying and asleep at the same time.⏎The question and answer are below.⏎Can we conclude from "Two young boys hold a crying infant while both are sitting in the same chair." that "The infant is fast asleep."?⏎Options:⏎- yes⏎- no⏎- i… [truncated, 1743 chars total] | Premise: "Woman skates in possession of puck."⏎Based on this premise, can we conclude that the hypothesis "The woman is figure skating." is true?⏎Options:⏎- yes⏎- it is not possible to tell⏎- no⏎no |

### Transformed — the same record (row 1 of the parquet for this exact (subset, task) pair)

One row of `data_clustered/flan/cot_fsopt_data__cot_esnli_ii.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| The man be showing his toys to adults and not just kids.. So what could be the question?⏎Question followed by answer: If "A man wearing dark green shirt and sweatpants is showing off a stuffed toy." does that mean that "A man is showing his stuffed toy to all the kids."?⏎Options:⏎- yes⏎- it is not possible to tell⏎- no⏎it is not possible to tell⏎⏎The infant cannot be crying and asleep at the same time.⏎The question and answer are below.⏎Can we conclude from "Two young boys hold a crying infant while both are sitting in the same chair." that "The infant is fast asleep."?⏎Options:⏎- yes⏎- no⏎- i… [truncated, 1743 chars total] | Premise: "Woman skates in possession of puck."⏎Based on this premise, can we conclude that the hypothesis "The woman is figure skating." is true?⏎Options:⏎- yes⏎- it is not possible to tell⏎- no⏎no | cot |

### Transformed — condition=`cot`

One row of `data_clustered/flan/cot_fsopt_data__cot_creak.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| **Q**⏎Is the following sentence factually correct?⏎"The Furry fandom shows up at many cosplay events."⏎Options:⏎- yes⏎- no⏎**A**⏎yes⏎Furry fandom does indeed go to cosplaying events, most of cosplay events feature some furries.⏎Is the following statement true?⏎"Alexandria Ocasio-Cortez serves as a California congresswoman."⏎no⏎Alexandria Ocasio-Cortez represents the state of New York.⏎**Q**⏎"Narayana Guru led a reform movement against the injustice in the caste-ridden society of Kerala in order to promote spiritual enlightenment and social equality."⏎Is the above claim true?⏎Options:⏎- yes⏎- n… [truncated, 1218 chars total] | no⏎Laughter is not a response of fear but to humor. | cot |

### Transformed — condition=`direct`

One row of `data_clustered/flan/dialog_fsopt_data__qrecc.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| Q: See the conversation. DIALOG:⏎What was Sun Ra's the Trip to Saturn about?⏎- ⏎****⏎Next: In 1936 or 1937 Sun Ra had a vision where he went to Saturn and aliens told him to drop out of college and make music.⏎⏎⏎Q: See the conversation. DIALOG:⏎What were wagenburg tactics used by Jan Zizka⏎- Jan Žižka helped develop tactics of using wagon forts, called vozová hradba in Czech or Wagenburg by the Germans, as mobile fortifications.⏎- How did wagenburg tactics help the Hussites win?⏎- When the Hussite army faced a numerically superior opponent they prepared carts for the battle by forming them int… [truncated, 3742 chars total] | Namie Amuro released eight number one singles on Oricon by the end of 1998.. | direct |

---

↑ [index](README.md) · ← [theoremqa](theoremqa.md) · [next → synth](synth.md)
