# Known issues / gotchas

← [README.md](README.md)

## uint8/uint16 dtype selection bug in `sample_tokenized.py` (fixed)

The sampling stage picks the dtype for the concatenated `tokens.npy` from
`vocab_size`. The original bound was:

```python
if vocab_size <= np.iinfo(np.uint16).max:   # 65536 <= 65535 -> False!
```

With vocab_size 65536 this silently fell back to **int32 — 2× memory and disk**
for the entire sampled dataset. Token ids live in `[0, vocab_size-1]`; we
verified no UNK/sentinel id exists anywhere in the pipeline (max token id =
65535 = uint16 max; `<|PAD|>` is a regular special token with id 0), so the
correct bound is `vocab_size - 1 <= dtype.max`.

Fixed: uint16 for vocab 65536, and an explicit `ValueError` beyond uint16
instead of silent int32. (The `TODO` comment was added by the repo owner in
commit `e4e62a6`; the bug itself dates to the authors' initial release
`c62331f`.)

## tokenizers-crate i32 pair-count overflow (present in BOTH tokenizers, on purpose)

The `tokenizers` 0.22 `BpeTrainer` keeps pair counts in `i32`
(`AHashMap<Pair, i32>`, wrapping release arithmetic) and only queues pairs
with `count > 0`. On this corpus `count("Ġ","t")` ≈ 3.46B and
`count("Ġ","a")` ≈ 2.63B — both exceed `i32::MAX` (2.15B) and wrap negative,
so those pairs are **never merged**: some ultra-frequent bigrams are simply
missing from the vocab. This affects the authors' artifact, the baseline run,
and — bug-for-bug, for exact parity — our iterative trainer
(`train_tokenizer_iter` documents and replicates the wrapping i32 semantics,
including the `count as u64` sign-extension in the priority queue and the
"fresh queue entries only for positive-change pairs" rule that keeps
overflowed pairs from ever re-entering the queue even after wrapping positive
again).

If retraining "more correctly", switch pair counts to u64 and expect a
slightly different (arguably better) vocab — but it will NOT match the
authors' artifact.

## Row-order sensitivity of per-file sampling

`prefix_config.yaml`-capped files are subsampled with `partial_shuffle` seeded
by `DefaultHasher(seed, safe_name)` over the file's **row order**. Our
streaming FLAN clean changed intra-file row order, so the sampled rows (and
hence the trained vocab) differ slightly from the authors' run: baseline vs
authors' artifact = 99.80% vocab overlap (65406/65536), merge common prefix
216 (measured on current artifacts; an earlier estimate quoted 4681 — that
does not reproduce against these files). Iterative vs baseline:
byte-identical (same words.bin → same merge sequence).

## Non-TTY progress bars

indicatif bars vanish in `docker logs`. Both trainers print explicit progress
lines instead (`[load]` every 15 s, `[train]` heartbeat, per-25-merge lines in
the iterative trainer). Don't "fix" the bars — read the lines.

## Resume must reconstruct queue eligibility (fixed)

A merges checkpoint stores only vocab + merges, so on resume the pair-count
priority queue is rebuilt from the replayed word state. Naively pushing all
pairs with count > 0 is WRONG on this corpus: pairs whose i32 count was
negative at init (overflow) but later wrapped positive through pure
destruction are never queued in an uninterrupted run (the reference only
queues pairs on positive *changes*), and resume would wrongly admit them.
Fix (both `train_tokenizer_iter` and `train_tokenizer_cpp`): on resume, count
pairs on the pre-replay state and admit a pair only if it was positive at
init or involves a merged token (id ≥ initial vocab size). Verified: SIGTERM
kill mid-run + resume produces byte-identical output for both trainers.

## Zombie-container teardown

After ~300G allocations the container teardown could hang in D-state
(uninterruptible kernel wait on memory reclaim/swap), leaving zombie
containers. Both wrapper scripts therefore use `docker run --init` (tini as
PID 1 reaps and signals cleanly). Keep `--init`.

## Why `config.json` says `{"model_type": "qwen3"}`

`train_tokenizer.rs` (and the iterative trainer) write a one-line
`config.json` next to `tokenizer.json` so HuggingFace `transformers` loads the
folder without complaining, using Qwen3 tokenizer conventions. The actual
tokenizer is a **from-scratch BPE trained on our data** — it is not a Qwen3
model and shares no vocab with it; the special-token *names* merely follow
Qwen conventions (see [special_tokens.md](special_tokens.md)).

## Round-trip testing note

`decode()` defaults to `skip_special_tokens=true`, which drops special tokens
from decoded text. When validating round-trips (`encode` → `decode`), pass
`skip_special_tokens=False` (Rust: `decode(ids, false)`) or use text without
special tokens. This affects any tokenizer, not just ours.
