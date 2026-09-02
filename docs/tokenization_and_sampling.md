# Tokenization + stratified sampling — design analysis

Scope: the README sections **Tokenization** and **(ON TRAINING NODES ONLY)
Stratified Sampling**, i.e. `tokenizer/src/main.rs` + `tokenizer/src/lib.rs`
(tokenize once → per-file `.npy` spans), `tokenizer/src/bin/train_tokenizer.rs`
(vocab training, where the marker-token ids come from) and
`sample_tokenized.py` + `prefix_config.yaml` (sample epochs into `/dev/shm`).
Analysis from code only; nothing here was executed.

## Research log — commands and sources used

Paper parsing (ocrc → local dots.mocr service, markdown + layout JSON per
page; see `AGENTS.md`):

```bash
# DeepSeek-V4 (single input + piped stdout streams the zip to stdout,
# so the bundle was unpacked manually from result_2606.tsv)
ocrc parse https://arxiv.org/pdf/2606.19348 --out /tmp/ocrc_papers \
  > /tmp/ocrc_papers/result_2606.tsv 2> /tmp/ocrc_papers/log_2606.txt

# batch with per-paper retries (DeepSeek-V3, R1, Qwen2.5, Qwen3, Kimi K2,
# Kimi k1.5, GLM-4.5, Llama 3)
for id in 2501.12948 2412.15115 2505.09388 2507.20534 2501.12599 \
          2508.06471 2407.21783; do
  for attempt in 1 2 3; do
    ocrc parse "https://arxiv.org/pdf/$id" --out /tmp/ocrc_papers \
      > "/tmp/ocrc_papers/result_$id.tsv" 2> "/tmp/ocrc_papers/log_$id.txt" \
      && break
  done
done

# bundles re-fetched by content hash when pipe mode skipped extraction:
curl -sS "http://127.0.0.1:8601/api/v1/documents/<sha256>/bundle?\
prompt_mode=prompt_layout_all_en" -o out.zip && unzip -d <dir> out.zip

# Llama 3: arxiv download kept timing out -> local PDF -> still fails:
curl -sSL --retry 5 --retry-all-errors -o llama3.pdf \
  https://arxiv.org/pdf/2407.21783
ocrc parse llama3.pdf --out /tmp/ocrc_papers     # parsing error (engine)
ocrc parse llama3.pdf --pages 0,1,...,25 ...     # same error
```

Known failures: **DeepSeek-R1** (2501.12948) and **Llama 3** (2407.21783)
crash the parsing engine (`parsing error`, persistent across retries; for
Llama 3 even a page-limited parse fails). Their facts below therefore come
from the HF `tokenizer_config.json` files, not the papers.

Tokenizer-config fact-checks (raw HF files):

```text
huggingface.co/Qwen/Qwen3-32B/raw/main/tokenizer_config.json
huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/tokenizer_config.json
huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/tokenizer_config.json
huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/tokenizer_config.json
huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/tokenizer_config.json
huggingface.co/zai-org/GLM-4.5/raw/main/tokenizer_config.json
huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/tokenizer_config.json  (mirror; meta-llama is gated)
github.com/xai-org/grok-1 (no tokenizer_config published; SentencePiece, 131072 vocab)
huggingface.co/Qwen/Qwen3-0.6B/raw/main/tokenizer_config.json  (Qwen3-32B extraction blanks some token contents; 0.6B used for the added-tokens table)
huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/tokenizer_config.json  (the tokenizer main.rs defaults point at)
huggingface.co/zai-org/GLM-5.2/raw/main/tokenizer_config.json
huggingface.co/zai-org/GLM-5.3/raw/main/tokenizer_config.json
huggingface.co/moonshotai/Kimi-K3/raw/main/tokenizer_config.json
```

Library semantics were checked against the vendored crate source:
`tokenizers-0.22.2/src/tokenizer/added_vocabulary.rs` (see §8.0).

## 0. Glossary

- **BOQ / EOQ** — *begin/end of question*: marker tokens wrapping the
  instruction span. Defaults reuse ChatML names: BOQ = `<|im_start|>`,
  EOQ = `<|im_end|>`.
- **EOA** — *end of answer*: the marker appended after the response. In this
  pipeline it should be `<|endoftext|>` (the EOS equivalent); the Rust
  binary's default is `<|box_end|>` — see §6.
- **`<|PAD|>`** — padding token, id 0. Currently unused: the sampler never
  emits it (each sample keeps its natural length; padding, if any, happens in
  the trainer, which is outside this repo).
- **condition tag vs condition token** — a *tag* is the string in the cleaned
  data (`"condition": "noisy,cot"` in the JSONL/parquet row). During
  tokenization each tag is mapped to a dedicated *token* in the vocab:
  `direct → <|direct|>` (id 1), `cot → <|cot|>` (2), `noisy → <|noisy|>` (3),
  `synth → <|synth|>` (4). These ids are **not** magic constants — they are
  the positions of those names in the `--special-tokens` list of
  `train_tokenizer.rs:52-68`; the tokenizer binary resolves them back by name
  via `token_to_id`.
- **task** — one tokenized source file = one output subdirectory
  (`<safe_name>`, path separators replaced by `__`). For clustered datasets
  one dataset = many tasks (`flan__cot__*` etc.); the report groups by the
  part before `__` ("category").
- **epoch** — one precomputed pass over the mix: a set of index arrays
  selecting which rows are trained on, in which order. See §2 for what is
  copied vs indexed.
- **DDP / rank / shard** — Distributed Data Parallel training: every node
  runs the same model and consumes a *shard* (slice) of each epoch; *rank* =
  the global index of a process. The sharding code lives in the trainer, not
  in this repo.
- **FS** — filesystem. "Shared FS" = a network filesystem visible to all
  nodes (NFS/Lustre/…); this pipeline deliberately avoids needing one at
  training time.
- **`/dev/shm`** — RAM-backed tmpfs on Linux. Writing the sampled dataset
  there = keeping it in memory, with a filesystem interface.
- **mmap** — memory-mapping a file so reads are served from the page cache /
  RAM instead of explicit I/O.

## 1. What the tokenization stage actually emits

Per cleaned row (`tokenizer/src/main.rs:231-255`):

```
instruction span: BOQ + <condition tokens...> + encode(instruction) + EOQ
response span:    encode(response) + EOA
```

- `inst_len` includes BOQ, the condition tokens and EOQ; `resp_len` includes
  the trailing EOA. This matters for every length computation downstream
  (see §5).
- Because the tags are real tokens inside the instruction span, the trained
  model is literally conditioned on the mix tags: prompt it with
  `<|cot|>` vs `<|direct|>` and you ask for different behavior.
- Per source file: `tokens.npy` (**u32** ids, concatenated) + four **u64**
  boundary arrays + `metadata.json` holding only `{source_mtime,
  source_size}`. Re-running the tokenizer skips files whose mtime+size are
  unchanged and deletes output dirs whose source disappeared ("prune
  orphans", `main.rs:151-162`) — that is the whole incremental story.
- `tokenizer_info.json` at the output root records the marker strings,
  condition mapping and vocab size; `sample_tokenized.py` reads the vocab
  size from it.

## 2. What the sampling stage does — one copy of the data, indices per epoch

`sample_tokenized.py` main flow:

1. Scan `tokenized_path/<task>/`, load the four index arrays per task, match
   the task name against `prefix_config.yaml` **prefixes in order, first
   match wins** (`:267-273`). Unmatched tasks silently get the default: no
   cap, `truncate`, `repeat=1`.
2. `concat_tokens` (`:86-131`): copy **every** task's full `tokens.npy`
   **exactly once** into one big array `output_path/tokens.npy`, recording
   each task's `mmap_base_offset`. dtype is uint8/uint16 by vocab size
   (65k → uint16; the bound bug here is fixed, see `tokenizer/docs/gotchas.md`).
   Concat runs **before** filtering (`:285-289`), so the array holds the
   whole corpus including rows about to be dropped — filtering only rewrites
   the index arrays, never the tokens.
3. `truncate_and_filter` per task (§5).
4. Epoch generation (`:292-346`): one Philox RNG seeded with `seed`. Per
   epoch, per task: `min(max_per_file, n_rows) * repeat` rows drawn via a
   permutation of the task (reshuffled on exhaustion → sampling without
   replacement per pass), spans translated to global offsets, one global
   permutation of the epoch buffer, then
   `epoch_N/{inst_start,inst_len,resp_start,resp_len}.npy` (**int64**) is
   written. `metadata.json` gets `max_seq_len` and per-epoch token total.
5. Markdown coverage report to stdout.

**No, the data is not copied per epoch.** Tokens exist in exactly one place
(the concat array); an epoch is *only* four int64 index arrays — 32 bytes per
row per epoch, versus ~600 bytes of actual token data per average row
(296.5 tokens × 2 bytes). Ten epochs of indices cost on the order of
tens of GiB, not ten copies of a 330 GiB corpus. The epochs are precomputed
(rather than generated on the fly) so that every node can load the identical,
already-shuffled schedule deterministically and so the trainer startup is
just an mmap.

**Why `max_per_file` / `repeat` exist.** Without caps, one epoch ≈ the raw
corpus mix: FLAN alone is 78% of all tokens, and small high-quality sets
would be noise. The YAML rebalances per epoch:

- `max_per_file: N` — at most N rows from *this file* per epoch (the cap is
  per file; a clustered dataset has many files). E.g. `dmmath__` is capped
  at 100k rows/file/epoch, `flan__` at 5k, `openmathinstruct2__` at 2M.
- `repeat: R` — every kept row is emitted R times per epoch (upsampling
  small HQ sets: Platypus-family, gsm8k, math_train etc. use `repeat: 10`).
- Effective rows per epoch per file: `min(cap, n_rows) * repeat`.

Rows beyond the cap are not lost — they enter later epochs through fresh
permutations, so multi-epoch training still covers the corpus.

## 3. The authors' logic — why it is built this way

- **Tokenize exactly once.** Boundaries are stored, not re-derived: the
  trainer can mask loss to the response span and the sampler can compute
  lengths by array indexing. The per-file layout keeps the tokenizer
  incremental.
- **Condition tags as tokens.** The mix semantics survive into the token
  stream itself, addressable at training/inference time.
- **Deterministic, materialized epochs.** A fixed-seed Philox produces the
  same bytes on every machine — the multi-node contract: each node runs the
  same script into its own `/dev/shm`, all nodes get an identical dataset,
  DDP ranks shard epochs consistently with zero coordination and no shared
  FS at training time.
- **`/dev/shm` is the point, not a quirk.** The trainer does random access
  into `tokens.npy` via the epoch indices; RAM turns the dataloader into
  pointer arithmetic. Cost: the corpus must fit in node RAM — 176.1B tokens
  × uint16 ≈ 330 GiB plus indices.
- **Mix control lives in YAML.** Caps/repeats rebalance the epoch without
  touching code; permutations give full coverage across epochs.

## 4. Multi-node → single-node: is a bias baked in?

Short answer: no stochastic bias, but two structural assumptions.

- Data is **bit-identical on every node** by construction (same seed, same
  permutations). One node changes nothing about what the model sees per
  epoch — if anything the per-epoch mix ratios hold *more* exactly, because
  a single node consumes whole epochs instead of sharding them across N
  nodes (within one shard the capped ratios hold only approximately).
- The genuine multi-node assumption is **"every node can hold the entire
  corpus in RAM"**. On one node the requirement stands: budget ≥ ~350 GiB
  for `/dev/shm` (or point `output_path` at an NVMe dir and accept
  page-cache-dependent reads).
- The only truly node-count-dependent piece is **how the trainer shards
  `epoch_N/` across ranks** — that code is not in this repo. Verify the
  trainer's per-rank slicing (stride vs contiguous) before assuming the
  README's sampling semantics carry over.
- Nothing in the sampler knows about node counts, world size or rank. The
  "multi-node bias" worry reduces to: same data, same order, plus the RAM
  requirement.

## 5. The truncation procedure — detailed look

`truncate_and_filter` (`sample_tokenized.py:68-83`), with
`context_size = 4097` (4096 + 1 AR shift):

```python
allowed_resp = context_size - min(inst_len, context_size)
# truncate mode (default):
keep if resp_len >= min_resp_length and allowed_resp >= 1
resp_len = min(resp_len, allowed_resp)
# drop mode:
keep if resp_len >= min_resp_length and resp_len <= allowed_resp
```

- **The instruction is never truncated.** Rows whose *instruction* alone
  reaches the context are **dropped entirely** in both modes
  (`min(inst_len, ctx)` caps at ctx → `allowed_resp = 0` → fails
  `allowed_resp >= 1`).
- **Only the response is cut, from the tail.** A truncated row keeps its
  first `allowed_resp` response tokens.
- **Truncated rows lose EOA.** EOA is the *last* token of the response span;
  capping `resp_len` cuts it off. The model is trained on targets that end
  mid-thought with no end marker — a direct attack on stop behavior. The
  share of rows without EOA equals the truncation rate.
- **The cut is at an arbitrary token boundary** — mid-word, mid-LaTeX,
  mid-code. The ids are valid, so nothing downstream complains; the damage
  is statistical.
- **The bias is systematic.** Truncation hits long responses —
  disproportionately the long-CoT sets (numinamath, acereason,
  openthoughts2, textbookreasoning, principia) — while short-answer sets
  (dmmath, sudoku, gsm8k) are untouched. The "how much explicit reasoning"
  knob is quietly turned down at the long, hard-reasoning end.
- **Your raw data is NOT corrupted.** Truncation rewrites only the index
  arrays in memory; per-file `tokens.npy` and the concat array are
  untouched. Re-running the sampler with different settings restores
  everything.
- **Edge case:** `allowed_resp >= 1` admits rows whose capped response is a
  single token (no EOA) — a one-token target that teaches almost nothing.
- **Length variability** below the cap is preserved (no padding/packing
  here); only the tail beyond 4097 is lost.

## 6. Time bombs in the codebase — what each one does and what it can cost

Each item: the mechanism, the failure mode, and the decision options.

### 6.1 Silent row loss in the tokenization stage

`main.rs:232-233` wraps encoding in `if let Ok(...) = tok.encode_fast(...)`.
A row that fails to encode is skipped — no counter, no log line. File-level
errors are printed but do not fail the run (`main.rs:186-189`), and the
progress bar increments either way.

- **Consequence:** the tokenized corpus can silently contain fewer rows than
  the cleaned corpus; nothing downstream compares the counts, so you only
  notice via a mysteriously smaller token total.
- **Options:** (a) accept — encode errors on already-validated UTF-8 are
  rare; (b) patch `process_row` to count skips per file and print a summary
  (small change, high diagnostic value); (c) at minimum, compare per-file
  row counts (`wc -l` / parquet metadata) against `len(inst_len.npy)` once.

### 6.2 The opposite policy in tokenizer *training*: panic on first bad file

`train_tokenizer.rs:134` calls `read_any_stream(...).unwrap()` inside a rayon
closure — one corrupt/truncated source file panics the whole training run
(in the iterative trainer this at least resumes from checkpoints).

- **Consequence:** tokenize-stage silently drops rows, train-stage dies on
  the same class of problem. Inconsistent failure policies mean you cannot
  predict behavior from one stage to the other.
- **Options:** leave as is (a loud crash during vocab training is arguably
  fine — you *want* to know); just don't "align" the two by making training
  silent.

### 6.3 Hardcoded marker defaults that alias the wrong tokens

`main.rs:37-46`: `--boq/--eoq/--eoa/--conditions` defaults target the stock
Qwen3 tokenizer, which has **no** dedicated condition tokens — so
`direct/cot/noisy/synth` default to Qwen *vision* tokens
(`<|object_ref_start|>`, `<|object_ref_end|>`, `<|quad_start|>`,
`<|quad_end|>`) and EOA defaults to `<|box_end|>`. With our trained tokenizer
the dedicated tokens exist, but nothing warns if you forget the flags.

- **Consequence:** a tokenized corpus where every condition marker is a
  vision-loc token and every answer ends with `<|box_end|>` instead of
  `<|endoftext|>` — semantically wrong conditioning and a stop token the
  trainer may not treat as EOS. Fully silent; only `tokenizer_info.json`
  records what was actually used.
- **Options:** (a) verify `tokenizer_info.json` of the artifact you sample
  from (condition_mapping + eoa fields) — one `cat`, do it now; (b) change
  the defaults in `main.rs` to the dedicated tokens so the safe path is the
  default path; (c) add a startup check: if the tokenizer has `<|direct|>`,
  require explicit `--conditions`.

### 6.4 Special-token ids are positional

Ids 0–30 are assigned by the order of the `--special-tokens` list in
`train_tokenizer.rs:52-68`. Retraining the vocab with a reordered/extended
list shifts every id.

- **Consequence:** anything keyed by id instead of name breaks — old
  tokenized datasets, trained checkpoints (embedding rows), external eval
  harnesses. Within this repo resolution is by name (`token_to_id`), so a
  *fresh* end-to-end run is self-consistent; mixing artifacts across vocab
  versions is not.
- **Options:** treat `tokenizer.json` + tokenized corpus + checkpoint as one
  versioned bundle; never reorder the list, only append.

### 6.5 u32 tokens on disk vs uint16 in the sampler

The tokenizer writes `tokens.npy` as **u32** (`main.rs:224`,
`Vec<u32>`); the sampler re-packs to **uint16** for vocab 65536
(`sample_tokenized.py:100-118`, fixed bound).

- **Consequence:** the intermediate tokenized corpus takes 4 bytes/token
  (~700 GiB for 176B tokens) where 2 would do — double the disk and double
  the read bandwidth during concat. Not a correctness issue (u32 is the
  safe default for arbitrary vocabs).
- **Options:** (a) accept — disk is cheap, and the final artifact is uint16;
  (b) emit uint16 directly from the tokenizer when
  `vocab_size - 1 <= u16::MAX` (here max id = 65535 fits exactly) — halves
  the intermediate footprint, costs a small patch and a re-tokenization.

### 6.6 Global RNG coupling

One Philox stream is consumed in sorted task order; adding/removing/renaming
a dataset changes the permutations of **all** datasets.

- **Consequence:** you cannot A/B a single dataset addition while holding
  the rest of the schedule fixed; "same seed" reproduces only the exact same
  task list.
- **Options:** accept (the epoch mix is what matters, not per-task order),
  or reseed per task with a hash of the task name if you need stable
  per-task draws.

### 6.7 Silent default config for unmatched prefixes

A task whose name matches no `prefix_config.yaml` entry gets `PrefixConfig()`
— no cap, truncate, repeat 1 — with no warning.

- **Consequence:** a typo'd prefix silently disables your intended cap/repeat
  and the corresponding dataset floods (or stops being upsampled in) the mix.
- **Options:** run the sampler once and read the coverage report — unmatched
  datasets show up with full row counts; or patch the loader to print the
  matched prefix per task.

### 6.8 Analytics are blind to truncation and drops

`Stats.cov_rows_truncated` is declared (`sample_tokenized.py:146`) but never
populated; the report shows coverage, not how many rows were dropped or
truncated by `truncate_and_filter`.

- **Consequence:** the most damaging knob of the pipeline (§5) is invisible
  in `show_analytics.md`; you can train for days on a corpus whose long-CoT
  tail was amputated without any number telling you.
- **Options:** populate `cov_rows_truncated` + a dropped-rows counter in
  `truncate_and_filter` (it has the masks in hand — trivial patch); or
  compute drop/truncate stats offline from the four index arrays alone (no
  token data needed).

## 7. Recommendations before a single-node training run

1. **Measure, don't guess.** Per dataset: rows dropped (instruction ≥ 4097),
   rows truncated, response-length percentiles vs 4097 — from the index
   arrays alone, or via the §6.8 patch.
2. **Prefer `drop` over `truncate` for HQ reasoning sets.** A missing
   solution teaches nothing; a solution cut mid-proof *without EOA* teaches
   "stop mid-thought". `long_context` is per-prefix — a YAML change, e.g.
   `drop` for numinamath/acereason/openthoughts2.
3. **If truncation is kept, consider restoring EOA** on truncated rows
   (overwrite the last kept response token with EOA at concat time, or emit
   a "has_eoa" mask). Absent in the authors' code — a deliberate quality
   trade worth revisiting.
4. **Verify the artifact's `tokenizer_info.json`** (§6.3): dedicated
   condition tokens and `<|endoftext|>` as EOA, not the Qwen-vision
   defaults.
5. **RAM budget:** `/dev/shm` ≥ ~350 GiB for the full 176B-token corpus at
   uint16, plus epoch buffers (§2). On one node there is no way around the
   concat array unless the sampler is rewritten to keep per-task files with
   (task, offset) indices.
6. **Keep the fixed seed workflow.** It costs nothing on one node and keeps
   runs comparable with the authors' setup.


## 8. Proposal: clean special-token design (full control assumed)

We control the whole pipeline and retraining the tokenizer is cheap (~1 h),
so the right move is not to work around the authors' hacks but to replace
them. This section describes the target design and the migration; nothing
here requires keeping any piece of the current special-token block.

### 8.0 What is actually wrong (recap)

- 31 special tokens, of which the pipeline uses ~7. The rest is the copied
  Qwen3 special-token block (vision/agent/FIM names). They can never enter
  our token stream: `main.rs` pushes only BOQ/condition/EOQ/EOA ids
  explicitly, and because `set_encode_special_tokens(true)` is set, the
  tokenizer *splits* special-token strings in corpus text as ordinary text
  instead of emitting their ids (tokenizers 0.22.2,
  `added_vocabulary.rs:163-164`: "Whether or not special tokens should be
  splitted when encoding. This is equivalent to ignoring them"). So the
  baggage is dead vocab, not a data hazard — dead vocab that costs embedding
  rows and, worse, invited the vision-token aliasing hack below.
- The CLI defaults alias `direct/cot/noisy/synth` onto Qwen *vision* tokens
  and EOA onto `<|box_end|>` — a silent wrong-semantics path (§6.3).
- Condition tokens are emitted **inside every training instruction**, so the
  model's input contract contains a hardcoded prompt chunk: arbitrary
  prompts at inference are out-of-distribution unless you hand-prepend magic
  tokens.

### 8.1 Target vocab: a minimal, honest special-token block

The recipe (no options, this is the list — rationale per token in §10 after
the cross-vendor survey):

```text
0 <|PAD|>            padding (trainer-side; never emitted by the data pipeline)
1 <|begin_of_text|>  BOS — first token of every training sequence
2 <|endoftext|>     EOS — terminates the response (the single stop token)
3 <|im_start|>       BOQ — opens the instruction block
4 <|im_end|>         EOQ — closes the instruction block
5 <|think|>          opens the reasoning block (cot rows)
6 </think|>          closes the reasoning block; answer follows
```

Nothing else. Every token carries a real, documented function; there is no
aliasing and no dead weight. Rules for keeping it sane forever:

- ids are positional — **append-only**, never reorder (§6.4);
- names say what the token *does*, not which model family inspired it;
- `vocab_size` stays 65536: fewer special tokens simply free a few merge
  slots for BPE (a rounding-level compression win).

**"Why is there an end token but no start token?"** — the question answers
itself once you see the current layout: every training sequence already
begins with a fixed marker (`<|im_start|>`), so BOQ has been the de-facto
start-of-work token all along; a separate BOS in front of it is redundant
for a pure-QA corpus (this is exactly Qwen3's design: `bos_token: null`,
`add_bos_token: false`, eos = `<|im_end|>`). The recipe still adds an
explicit `<|begin_of_text|>` because DeepSeek-V3/R1/V4
(`<｜begin▁of▁sentence｜>`), Llama 3 (`<|begin_of_text|>`) and Kimi K2
(`[BOS]`) all carry one: it costs one embedding row, gives generation an
unambiguous anchor token, and keeps the door open for non-QA pretraining
data where no `im_start` is present. Sequence layout:
`BOS BOQ inst EOQ [think … /think] answer EOS`.

**No `<|answer|>` marker.** The reference thinking models (DeepSeek-R1/V4,
Qwen3) do not use one: the final answer is simply the text after the
closing think tag. A dedicated answer token would add a third nesting level
for zero gain; evaluation extracts the post-`</think|>` span instead.

### 8.2 Target token stream: no conditioning on mix metadata

```text
cot rows:    <|begin_of_text|> <|im_start|> instruction <|im_end|> <|think|> reasoning </think|> answer <|endoftext|>
direct rows: <|begin_of_text|> <|im_start|> instruction <|im_end|> answer <|endoftext|>
```

- **No condition tokens in the stream.** `direct/cot/noisy/synth` are
  dataset metadata: they live in the cleaned `condition` column (for
  analytics and filtering) and in `prefix_config.yaml` (for mix control via
  file prefixes). Neither consumer needs the model to see them.
- The cot/direct distinction is expressed the way ordinary thinking models
  express it: reasoning wrapped in think-tags, final answer after them, one
  EOS. At inference the model thinks by default, or is nudged to a direct
  answer by prefilling an empty `<|think|></think|>`. Arbitrary prompts are
  in-distribution: `<|im_start|>prompt<|im_end|>` → generate until
  `<|endoftext|>`.
- Loss masking is unaffected: the trainer still gets instruction/response
  spans (the response span now merely contains think-tags inside).

Honest cost: most cleaning scripts emit the full solution as `response` and
the short answer as a *separate* `direct` row. To build
`reasoning | answer` for cot rows, reuse the paired direct rows (join on
instruction text — clean for math_train/numinamath/omnimath, which have
explicit short answers) or wrap the whole solution in think-tags and append
nothing (weaker "answer after think" pattern). This is the only step that
touches per-dataset logic.

### 8.3 Migration plan

1. **Vocab**: shrink the `--special-tokens` list in
   `train_tokenizer.rs:52-68` (and the iterative trainer) to §8.1, retrain
   (~1 h; the iterative trainer checkpoints, so this is restartable).
   Nothing else about BPE training changes.
2. **Tokenization stage** (`main.rs`): delete the condition-token emission
   from `process_row`; make EOA default to `<|endoftext|>`; delete the
   `--conditions` flag and the vision-token aliasing path outright (a
   removed code path cannot be silently wrong). If doing §8.2's
   think-format, wrap cot responses here (the condition string is already
   passed to `process_row`).
3. **Re-tokenize the corpus** into a fresh output dir (the incremental
   mtime/size machinery is irrelevant for a clean rebuild; keep the old
   artifact until the new one is verified).
4. **Re-run the sampler** — unchanged code; it reads spans and vocab size,
   not marker names.
5. **Verify**: `tokenizer_info.json` shows the new mapping; per-file row
   counts match the cleaned corpus (§6.1 check); drop/truncate stats per
   §7.1 before any training.

### 8.4 What this costs and what it breaks

- Costs: ~1 h vocab retrain + one full re-tokenization + one sampler run.
- Breaks (intentionally): comparability with the authors' vocab/artifacts —
  the parity notes in `tokenizer/docs/gotchas.md` (99.8% vocab overlap, the
  i32-overflow bug-for-bug replication) stop applying the moment you change
  the special-token list or fix the overflow. If you ever need to reproduce
  their numbers, keep their artifact frozen; otherwise let it go.
- Does NOT break: the cleaning pipeline, the sampler, the epoch/index
  format, the training-side contract (spans + max_seq_len + uint16).

### 8.5 Decision summary

The authors' format is a demo-day shortcut, not a constraint. Recommended
end state: §8.1 vocab + §8.2 stream + §7 truncation policy. Each is an
independent, revertible change; together they remove every silent-semantics
path found in §6.

## 9. Can one standard format cover all 24 datasets without quality loss?

Question: does the §8 format (`BOQ inst EOQ [<|think|> reasoning </|think|>
answer] EOA`, no condition tokens) fit every dataset in
`scripts/docs/README.md`, or do some rows lose something when forced into it?

The datasets fall into four shapes:

1. **Reasoning + extractable short answer** — math_train (`\boxed{}`),
   numinamath (`answer` field, non-proof rows), omnimath, scibench,
   scienceqa (rationale + letter), ampsmathematica, gsm8k (`####`). These map
   cleanly: solution → think block, short answer → post-think answer. No
   loss; the format actually *adds* structure the current flat `response`
   lacks.
2. **Reasoning trace without a separable answer** — acereason and
   openthoughts2 (R1-style distilled traces, answer embedded in the trace),
   principia (synthetic STEM drill), textbookreasoning, natural_reasoning,
   webinstruct_verified, amps_khan (hint sequences). Options: wrap the whole
   response in think-tags (answer position stays empty) or run a cheap
   extraction pass. Either way nothing is *lost* — worst case the
   "answer after think" pattern is absent for those rows, which is also what
   proof rows in numinamath legitimately look like.
3. **Direct answers** — dmmath, sudoku_extreme, gsm8k-direct rows, MCQ sets
   (openbookqa, reclor, scienceqa-direct, arb-law), theoremqa, tasksource,
   flan-direct, no_robots, synth-direct. Trivial fit: `BOQ inst EOQ answer
   EOA`. No think block — matches the standard "empty think" behavior of
   thinking models run in non-thinking mode.
4. **FLAN cot / synth cot subsets** — CoT text exists but with no marked
   final answer; same treatment as group 2.

Format-level risks and their sizes:

- **Marker overhead on tiny rows.** dmmath rows average ~40 tokens; the
  4-token frame (BOQ/EOQ/EOA + think tags when present) is already in the
  current pipeline (it emits BOQ/conditions/EOQ/EOA too), so the change is
  ±0–2 tokens per row. Nothing to optimize.
- **Loss of the condition-token lever.** Replaced by the think-format lever
  (think vs empty-think) plus prefix-level mix control — see §8.2. The only
  genuinely lost ability is "generate with a *noisy* prior", which has no
  conceivable use.
- **HRM-specific structure.** The inst/resp spans survive unchanged (the
  response span simply contains think-tags), so nothing in the
  span/loss-masking contract changes.
- **Evaluation.** Decontamination and benchmark mapping
  (`scripts/docs/evaluation.md`) are format-independent.

Verdict: yes — one standard format covers all 24 datasets; the only real
work item is answer extraction for group 2, and its fallback (think-wrap
whole response, no answer) costs no data and matches how proof rows look
anyway.

## 10. Cross-vendor survey: how the majors handle tokenizers and special tokens

Sources: the parsed papers listed in the research log (page cites are the
arXiv PDF pages) plus the raw `tokenizer_config.json` files linked there.
HF config facts are stated without page numbers — the config *is* the source.

### DeepSeek (V3, R1, V4)

- Byte-level BPE, 128K vocab; the pretokenizer was retuned for multilingual
  compression and introduces merged punctuation+line-break tokens; to fight
  the resulting token-boundary bias they randomly split a share of those
  merged tokens during training (DeepSeek-V3, §4.1, PDF p. 22).
- V4 keeps the V3 tokenizer and 128K vocab, adding only "a few special
  tokens for context construction" (DeepSeek-V4, §4.1, p. 24).
- BOS `<｜begin▁of▁sentence｜>`, EOS `<｜end▁of▁sentence｜>`, pad = EOS; chat
  roles `<｜User｜>` / `<｜Assistant｜>` (HF: DeepSeek-V3
  tokenizer_config.json).
- R1's chat template ends the assistant turn with EOS and prefills
  generation with `<｜Assistant｜><think>\n`; in R1 the think tags are
  literal template text (HF: DeepSeek-R1 tokenizer_config.json).
- V4 makes `<think></think>` a *dedicated* tag pair (DeepSeek-V4, §5.1.1,
  pp. 28–31) and adds a `|DSML|` special token for an XML-based tool-call
  schema (ibid., Table 4).
- **Quick Instruction** (DeepSeek-V4, §5.1.1, Table 5, ~p. 30): dedicated
  special tokens (`<|action|>`, `<|title|>`, `<|query|>`, `<|authority|>`,
  `<|domain|>`, `<|extracted_url|>`, `<|read_url|>`) appended to the input
  to trigger auxiliary tasks (search-trigger decision, title/query
  generation, domain classification) while reusing the KV cache. Note the
  division of labor: these tokens are appended by the *serving stack* per
  request — they are not baked into every pretraining document. This is the
  clean version of what the HRM-Text authors did dirtier with condition
  tokens (§8.0).

### Qwen (Qwen2.5, Qwen3)

- BBPE, 151,643 regular tokens in Qwen2.5; the control-token set was
  expanded from 3 to 22 (two of the new ones for tool calls), unifying the
  vocabulary across all Qwen2.5 sizes (Qwen2.5, §2 "Architecture &
  Tokenizer", PDF p. 3).
- Qwen3 keeps the same tokenizer, vocabulary 151,669 (Qwen3, §2, PDF p. 3).
- ~151.6K vocab BBPE; **no BOS** (`bos_token: null`, `add_bos_token:
  false`); eos = `<|im_end|>`; pad = `<|endoftext|>`; ChatML roles
  `<|im_start|>` / `<|im_end|>` (HF: Qwen3-32B tokenizer_config.json).
- Thinking/non-thinking is a **template-level** mechanism, trained in the
  Thinking Mode Fusion SFT stage: `/think` and `/no_think` flags go into the
  user query (plain text, not special tokens), and non-thinking samples keep
  an **empty think block** in the assistant response so the output format
  stays identical across modes (Qwen3, §post-training, Table 9, PDF
  pp. 11–12). `enable_thinking=false` in the chat template just concatenates
  that empty think block (HF config). "Thinking budget" is bolted on at
  inference: when the thinking length hits the budget, a fixed stop-thinking
  instruction plus `</think>` is injected and the model answers from the
  partial reasoning — an emergent ability, not a trained one (Qwen3, PDF
  p. 11).
- `<think>`/`</think>` **are** in Qwen3's added-tokens table (ids
  151667–151668, `special: false`) — verified on the Qwen3-0.6B config (the
  Qwen3-32B config extraction blanks these entries; do not trust it). The
  chat template still treats them as literal text inside assistant content;
  `enable_thinking=false` inserts an empty `<think>\n\n</think>\n\n` block
  (HF: Qwen3-0.6B tokenizer_config.json).

### Kimi K2 (Moonshot)

- tiktoken-family tokenizer, ~163.8K vocab; explicit `[BOS]` / `[EOS]` /
  `[UNK]` / `[PAD]`; role tokens `<|im_user|>`, `<|im_assistant|>`,
  `<|im_system|>`, `<|im_middle|>`; tool-call section tokens (HF:
  Kimi-K2-Instruct tokenizer_config.json). The K2 and k1.5 papers themselves
  do not document the tokenizer at all — the config is the only source.

### GLM-4.5 (Z.ai)

- ~151.3K vocab; eos = pad = `<|endoftext|>`; roles `<|system|>` /
  `<|user|>` / `<|assistant|>` / `<|observation|>`; a dedicated `/nothink`
  special token switches reasoning off; legacy `[MASK]`/`[gMASK]`/`[sMASK]`
  kept (HF: GLM-4.5 tokenizer_config.json). The GLM-4.5 paper likewise does
  not document the tokenizer.

### Llama 3.1 (Meta)

- 128K BPE + 256 reserved special-token slots (ids 128000–128255); BOS
  `<|begin_of_text|>`, EOS `<|end_of_text|>`, dedicated pad
  `<|finetune_right_pad_id|>`; structure tokens `<|start_header_id|>` /
  `<|end_header_id|>` / `<|eot_id|>` (HF: Meta-Llama-3.1-8B
  tokenizer_config.json, mirror).

### Grok-1 (xAI)

- SentencePiece, 131,072 vocab; released as a base model with no published
  chat/special-token layout — `xai-org/grok-1` ships no
  `tokenizer_config.json` at all (github.com/xai-org/grok-1).

### The mid-2026 generation (GLM-5.x, Kimi K3, Qwen3-Next)

Configs checked directly; the point of this subsection is whether the new
flagships changed the conventions above. They did not — they converged
further:

- **GLM-5.2 / GLM-5.3** (Z.ai, Aug 2026): same GLM block as 4.5 —
  eos = pad = `<|endoftext|>`, roles `<|system|>/<|user|>/<|assistant|>/<|observation|>`,
  multimodal `begin/end_of_*` markers; 5.3 re-adds `<sop>`/`<eop>`; 1M
  context (HF: GLM-5.2 and GLM-5.3 tokenizer_config.json).
- **Kimi K3** (Moonshot, Aug 2026): keeps the tiktoken-family tokenizer and
  the `[BOS]`/`[EOS]`/`[UNK]`/`[PAD]` core, but the chat markers moved to a
  header style: `[start_header_id]`/`[end_header_id]` +
  `<|end_of_msg|>` + `[EOT]`, plus `<|media_begin|>/<|media_content|>/<|media_end|>/<|media_pad|>`
  (HF: Kimi-K3 tokenizer_config.json).
- **Qwen3-Next** — notable because it is the tokenizer this repo's
  `main.rs` defaults point at: the ids block 151643–151668 matches Qwen3
  (vision tokens included), `<tool_call>`/`<tool_response>`/`<think>` pairs
  are added tokens (special: false), no BOS, eos = `<|im_end|>` (HF:
  Qwen3-Next-80B-A3B-Instruct tokenizer_config.json).

Takeaway: as of mid-2026 the field still runs on the same skeleton — one
EOS that ends the assistant turn, ChatML-style role delimiters, think-tags,
a dedicated or EOS-aliased pad — and vendors mostly *rename* the same
handful of roles rather than invent new structure. The §8.1 recipe sits
exactly on this converged skeleton.

### Cross-vendor takeaways (and how §8.1 follows from them)

1. **BOS**: present in DeepSeek/Llama/Kimi, absent in Qwen/GLM. Both designs
   are proven at scale; §8.1 includes `<|begin_of_text|>` for an explicit
   generation anchor and non-QA future data.
2. **One EOS doubles as end-of-answer** everywhere: DeepSeek terminates each
   assistant turn with EOS; Qwen's eos *is* the turn-closing `<|im_end|>`.
   §8.1 keeps a single `<|endoftext|>` stop token.
3. **PAD**: only Llama and Kimi bother with a dedicated pad token;
   DeepSeek/Qwen/GLM pad with EOS. §8.1 keeps `<|PAD|>` separate so padding
   can never be confused with a real stop signal in the loss mask.
4. **Think tags**: present as added tokens in Qwen3 (ids 151667–151668) and
   a dedicated tag pair in DeepSeek-V4; R1's template treats them as literal
   text. §8.1 makes `<|think|>`/`</think|>` real special tokens —
   atomicity guaranteed by the vocab, and corpus text cannot mint them
   (§8.0).
5. **Aux-task tokens are a serving-layer pattern** (DeepSeek-V4 Quick
   Instruction): appended per request by the system, never baked into the
   pretraining stream. This is precisely why the HRM-Text authors' condition
   tokens inside every training row are the wrong layer — and why §8.2
   removes them from the token stream entirely.
