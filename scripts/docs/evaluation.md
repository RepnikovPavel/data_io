# Evaluation & benchmarks — relation to the training-data pipeline

↑ [index](README.md)

How the benchmarks in the HRM-Text paper (arXiv:2605.20613) relate to the
datasets processed by this repo, and how accuracy is computed given that the
training data was transformed. Paper facts are quoted from the paper; pipeline
claims are verified against the scripts in `pipe/`, `pipe/clean_platypus/`,
`pipe_clustered/`.

## The benchmarks

Table 4 of the paper evaluates: **MMLU, ARC-C, HellaSwag, Winogrande, BoolQ,
DROP, GSM8K, MATH** — against Llama/Qwen/Gemma/OLMo/Huginn/Ouro baselines.

## Key point: eval does not consume the transformed data

Benchmark accuracy is computed from the **original benchmark repositories**
(test/validation splits), prompted in the original format. The scripts in this
repo only repackage *training-side* data into `(instruction, response,
condition)` records for pretraining; they never produce eval prompts or gold
answers. **The transform therefore has no effect on how accuracy is computed —
only on what the model saw during training.**

One intentional coupling exists (paper §4.1): the model is trained with
condition tags, and at inference a condition tag is prepended to the
instruction to select the response style ("four primary conditions: direct
(answer-only), cot (chain-of-thought), synth, noisy"). The tags in the
`condition` column documented per dataset are exactly this mechanism.

## Eval protocol (paper Table 8 and §4.3)

| setting | value |
|---|---|
| max context | 3072 tokens |
| decoding | temperature 0 (deterministic) |
| system prompt | none |
| prompt contents | original benchmark question only; few-shot exemplars only where the benchmark protocol requires them |
| few-shot evals | vLLM |
| chain-of-thought evals | `lm_eval_harness` |
| checkpoint | EMA of weights (decay 0.9999) — used for the final evals and the released weights |

## Split separation: what is trained on vs. what is evaluated

| training dataset (this repo) | in the training mix | relation to Table 4 eval |
|---|---|---|
| gsm8k_train | GSM8K **train** split (7,473 rows) | GSM8K **test** (1,319) is evaluated — disjoint by construction |
| math_train | MATH **train**, 7 subjects (14,996 records) | MATH **test** (5,000) is evaluated — disjoint |
| openmathinstruct2, acereason | synthetic solutions derived from GSM8K/MATH **train** problems only | same train/test separation holds upstream |
| flan | train splits of many NLP benchmarks — verified in the output: `flan_*__ai2_arc_*`, `__drop_*`, `__hellaswag_*`, `__winogrande_*`, `*boolq*` task files (ARC-C/E, DROP, HellaSwag, Winogrande, BoolQ) | eval uses those benchmarks' test/validation splits; standard FLAN practice; the eval *questions* are covered by the n-gram decontamination test below |
| tasksource | train splits of 182 curated NLP tasks | none of the Table 4 benchmarks directly; same decontamination coverage |
| omnimath | Omni-MATH **test** split is used as *training* data — deliberate: Omni-MATH is not among the Table 4 benchmarks | **flag:** do not add Omni-MATH to the eval suite without removing it from the mix |
| theoremqa, scibench, reclor, arb, openbookqa, scienceqa | benchmarks repurposed as training data (Platypus collection): TheoremQA test, ReClor train+val, OpenBookQA train+val, ScienceQA all splits, SciBench full | none are Table 4 benchmarks; **flag:** evaluating on any of them requires excluding it from training first |
| synth, dmmath, ampsmathematica, amps_khan, sudoku_extreme, principia_collection, natural_reasoning, webinstruct_verified, no_robots, numinamath, openthoughts2, textbookreasoning | synthetic, procedural, or web-derived corpora | no direct benchmark split relationship; covered by the decontamination test |

## Decontamination (paper §4.2)

Method (adapted from the Llama family): tokenize the questions of **all
evaluated benchmarks** (excluding few-shot exemplars) and find **n-gram
matches (n = 13 and n = 20) against the fully tokenized pretraining corpus**.
A sample's contamination % is the fraction of its tokens inside matched
n-grams. Eval samples are partitioned into Clean (<20%), Not Clean (≥20%),
Not Dirty (<80%), Dirty (≥80%); contamination is deemed significant only if
|Z| > 2 on all four subsets (clean must do worse, dirty better).

Result: HRM-Text 0.6B — no significant contamination at either n. HRM-Text 1B
— significant only on DROP at n = 13 (not at n = 20), and it still scores
81.1 on the strictly-clean DROP subset (0% contamination, 5,904 samples).
Conclusion: benchmark performance is unlikely to be driven by exposure to
test examples.

## Practical rules for extending the recipe

1. **Before adding a benchmark to eval**, check it is not in the training
   mix: this index plus the split table above, and for aggregate mixtures
   (FLAN, tasksource) check the per-task output files —
   `ls /mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/flan | grep -i <benchmark>`.
2. **Run the paper's n-gram decontamination test** (n = 13 and n = 20,
   benchmark questions vs. the tokenized corpus, clean/dirty subset analysis)
   for any new benchmark.
3. **Never train on the eval split** of an active benchmark. Training on a
   non-evaluated benchmark's test split (as done for Omni-MATH) is a
   deliberate one-way decision: document it in that dataset's doc page.
