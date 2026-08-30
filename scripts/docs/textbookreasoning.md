# textbookreasoning

**Script:** `pipe_clustered/clean_textbookreasoning.py`

## Purpose

TextbookReasoning (MegaScience): QA extracted from textbooks. Every row goes to cot.parquet (full answer); non-proof rows also go to direct.parquet (short reference answer).

## Before (raw storage)

- Source: HF `MegaScience/TextbookReasoning` (split `train`)
- Format: HF dataset (arrow) in the prefetched local cache

Columns actually read by the transform:

| column | type | meaning |
|---|---|---|
| `question` | string | the question |
| `answer` | string | full answer (cot.parquet response) |
| `reference_answer` | string | short answer (direct.parquet response; 'prove'/'show that' questions excluded) |

## After (transformed)

- Location: `data_clustered/textbookreasoning/cot.parquet`, `data_clustered/textbookreasoning/direct.parquet` (under `/mnt/hdd2/datasets_text_transformed/HRM-Text`; 2 file(s))
- Format: parquet (snappy) — columns `instruction` (string), `response` (string), `condition` (string)
- Rows: 1,178,449

`condition` values used here:

- `synth,cot` — cot.parquet: full extracted answer
- `noisy,direct` — direct.parquet: short reference answer

## Examples

Fields longer than 600 chars are truncated (`… [truncated, N chars total]`).

### Raw record (HF `MegaScience/TextbookReasoning`, split `train`)

````text
{
  "question": "In a patient diagnosed with a testicular germ cell tumor, what does the presence of elevated HCG levels without elevated AFP indicate?",
  "answer": "Elevated HCG levels without elevated AFP are strongly suggestive of a seminoma. This is because seminomas typically produce HCG but not AFP, whereas nonseminomatous germ cell tumors (e.g., embryonal carcinoma or yolk sac tumor) often produce AFP. Other markers, such as CD30 or cytokeratin, are associated with nonseminomatous components. Thus, the absence of AFP helps differentiate seminoma from other germ cell tumor subtypes.  \n\n$\\boxed{\\text{Suggestive of a seminoma.}}$",
  "reference_answer": "Suggestive of a seminoma."
}
````

### Transformed record (`data_clustered/textbookreasoning/cot.parquet`, record 1)

````text
{
  "instruction": "In a patient diagnosed with a testicular germ cell tumor, what does the presence of elevated HCG levels without elevated AFP indicate?",
  "response": "Elevated HCG levels without elevated AFP are strongly suggestive of a seminoma. This is because seminomas typically produce HCG but not AFP, whereas nonseminomatous germ cell tumors (e.g., embryonal carcinoma or yolk sac tumor) often produce AFP. Other markers, such as CD30 or cytokeratin, are associated with nonseminomatous components. Thus, the absence of AFP helps differentiate seminoma from other germ cell tumor subtypes.  \n\n$\\boxed{\\text{Suggestive of a seminoma.}}$",
  "condition": "synth,cot"
}
````

### Transformed record (`data_clustered/textbookreasoning/cot.parquet`, record 2)

````text
{
  "instruction": "Let $\\vec{u}=\\langle 3,1\\rangle$, $\\vec{v}=\\langle-2,6\\rangle$, and $\\vec{w}=\\langle-4,3\\rangle$. Find the angles $\\alpha$, $\\beta$, and $\\theta$, where:\n- $\\alpha$ is the angle between $\\vec{u}$ and $\\vec{v}$,\n- $\\beta$ is the angle between $\\vec{v}$ and $\\vec{w}$,\n- $\\theta$ is the angle between $\\vec{u}$ and $\\vec{w}$.",
  "response": "To find the angles, we use the dot product formula:\n\\[\n\\theta = \\cos^{-1}\\left(\\frac{\\vec{a} \\cdot \\vec{b}}{\\|\\vec{a}\\|\\|\\vec{b}\\|}\\right)\n\\]\nwhere $\\vec{a}$ and $\\vec{b}$ are the vectors in question.\n\n1. **Compute the magnitudes of the vectors**:\n   \\[\n   \\|\\vec{u}\\| = \\sqrt{3^2 + 1^2} = \\sqrt{10}, \\quad \\|\\vec{v}\\| = \\sqrt{(-2)^2 + 6^2} = 2\\sqrt{10}, \\quad \\|\\vec{w}\\| = \\sqrt{(-4)^2 + 3^2} = 5.\n   \\]\n\n2. **Compute the dot products**:\n   \\[\n   \\vec{u} \\cdot \\vec{v} = 3(-2) + 1(6) = 0, \\quad \\vec{v} \\cdot \\vec{w} = (-2)(-4) + 6(3) = 26, \\quad \\vec{u} \\cdot \\vec{w} = 3(-4) + 1(3) = -9.\n   \\]\n\n3\n… [truncated, 1419 chars total]",
  "condition": "synth,cot"
}
````
