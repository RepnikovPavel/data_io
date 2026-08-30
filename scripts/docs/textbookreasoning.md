# textbookreasoning

↑ [index](README.md) · ← [tasksource](tasksource.md)

**Script:** `pipe_clustered/clean_textbookreasoning.py`

## Purpose

TextbookReasoning (MegaScience): 650k questions with truthful reference answers extracted from 12k university textbooks across 7 scientific disciplines. Every row goes to cot.parquet (full answer); non-proof rows also go to direct.parquet (short reference answer).

## Before (raw storage)

- Source: HF `MegaScience/TextbookReasoning` (split `train`)
- Storage: HF dataset (arrow table) in the prefetched local cache

Fields read by the transform (types derived from the actual files):

| field | type | meaning |
|---|---|---|
| `question` | UTF-8 text (arrow `string`) | the question |
| `answer` | UTF-8 text (arrow `string`) | full answer (cot.parquet response) |
| `reference_answer` | UTF-8 text (arrow `string`) | short answer (direct.parquet response; 'prove'/'show that' questions excluded) |

## After (transformed)

- Location: `data_clustered/textbookreasoning/cot.parquet`, `data_clustered/textbookreasoning/direct.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 2 file(s))
- Storage: parquet; columns `instruction`, `response`, `condition`
- Rows: 1,178,449

| column | type | meaning |
|---|---|---|
| `instruction` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the prompt |
| `response` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | the target |
| `condition` | parquet BYTE_ARRAY (logical String) → arrow `string` — UTF-8 text | comma-separated tags (see keyword table below) |

## Keyword: `condition` (output)

Every output record carries this tag; training samples/mixes by it.

| value | rows | meaning | why it matters |
|---|---|---|---|
| `synth,cot` | 651,840 | cot.parquet: full extracted answer | textbook-grade scientific reasoning |
| `noisy,direct` | 526,609 | direct.parquet: short reference answer | answer-only variant for verifiable-style training |

_exact counts (condition is constant per file; rows summed from parquet metadata)_

## Keyword: `subject` (input)

Scientific discipline (unused by the transform; shown for context).

| value | rows |
|---|---|
| `math` | 19,366 |
| `medicine` | 3,786 |
| `biology` | 2,423 |
| `physics` | 2,024 |
| `chemistry` | 1,523 |
| `cs` | 803 |
| `economics` | 75 |

_estimated from the first 30,000 of 651,840 rows_

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`); in tables, `⏎` marks a newline inside the value.

### Raw — HF `MegaScience/TextbookReasoning`, split `train`

One row of HF `MegaScience/TextbookReasoning`, split `train` (an arrow table in the local HF cache), shown as a table with the columns the transform reads:

| question | answer | reference_answer |
|---|---|---|
| In a patient diagnosed with a testicular germ cell tumor, what does the presence of elevated HCG levels without elevated AFP indicate? | Elevated HCG levels without elevated AFP are strongly suggestive of a seminoma. This is because seminomas typically produce HCG but not AFP, whereas nonseminomatous germ cell tumors (e.g., embryonal carcinoma or yolk sac tumor) often produce AFP. Other markers, such as CD30 or cytokeratin, are associated with nonseminomatous components. Thus, the absence of AFP helps differentiate seminoma from other germ cell tumor subtypes.  ⏎⏎$\boxed{\text{Suggestive of a seminoma.}}$ | Suggestive of a seminoma. |

### Transformed — condition=`synth,cot`

One row of `data_clustered/textbookreasoning/cot.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| In a patient diagnosed with a testicular germ cell tumor, what does the presence of elevated HCG levels without elevated AFP indicate? | Elevated HCG levels without elevated AFP are strongly suggestive of a seminoma. This is because seminomas typically produce HCG but not AFP, whereas nonseminomatous germ cell tumors (e.g., embryonal carcinoma or yolk sac tumor) often produce AFP. Other markers, such as CD30 or cytokeratin, are associated with nonseminomatous components. Thus, the absence of AFP helps differentiate seminoma from other germ cell tumor subtypes.  ⏎⏎$\boxed{\text{Suggestive of a seminoma.}}$ | synth,cot |

### Transformed — condition=`noisy,direct`

One row of `data_clustered/textbookreasoning/direct.parquet` (parquet table, columns: `instruction`, `response`, `condition`), shown as a table:

| instruction | response | condition |
|---|---|---|
| In a patient diagnosed with a testicular germ cell tumor, what does the presence of elevated HCG levels without elevated AFP indicate? | Suggestive of a seminoma. | noisy,direct |

---

↑ [index](README.md) · ← [tasksource](tasksource.md)
