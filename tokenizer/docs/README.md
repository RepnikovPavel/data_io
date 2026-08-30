# Tokenizer stage

Training the HRM-Text BPE tokenizer: byte-level BPE, vocab_size **65536**, NFC
normalizer, GPT-2-style pre-tokenizer regex, 31 special tokens (ids 0..30).
Rust crate: `tokenizer/` (see `tokenizer/Cargo.toml`).

Subpages:

- [pipeline.md](pipeline.md) — how to run: download → clean → stage to NVMe →
  train; inputs/outputs; checkpoint & resume semantics of the iterative trainer
- [benchmarks.md](benchmarks.md) — measured timings/memory (48 cores, 251G RAM)
  and a cloud cost estimate
- [special_tokens.md](special_tokens.md) — all 31 added tokens: id, purpose in
  this project, origin
- [gotchas.md](gotchas.md) — known issues: the uint16 dtype bug, the
  tokenizers-crate i32 overflow, row-order sensitivity, non-TTY progress bars,
  container teardown
- [comparison.md](comparison.md) — authors' artifact vs baseline vs iterative:
  measured equality numbers

## TL;DR

Two trainers, one output format:

| | `train_tokenizer` (baseline) | `train_tokenizer_iter` (ours) |
|---|---|---|
| Source | `tokenizer/src/bin/train_tokenizer.rs` | `tokenizer/src/bin/train_tokenizer_iter.rs` |
| Algorithm | `tokenizers` crate `BpeTrainer` (authors' algorithm) | own incremental BPE loop, bug-for-bug parity |
| Phases | one-shot | `load` (→ `words.bin`) + `train` (→ `merges_N.bin` checkpoints), resumable |
| Total time | ~92 min | ~35 min |
| Peak memory | ~280G virt (+ ~220G NVMe swap spill) | ~65G RAM, no swap |
| Result | `original/bpe/tokenizer.json` | **byte-identical** to baseline ([proof](comparison.md)) |

Artifacts land in `/mnt/hdd2/models/HRM-Text/tokenizers/{original,iterative}/bpe/`
(`tokenizer.json` + `config.json`).

Quick run (data already staged on NVMe):

```sh
./scripts/prepare_tokenizer_data.sh   # once: rsync HDD -> NVMe
./scripts/train_tokenizer_iter.sh     # load (~30 min, once) + train (~5 min)
```

The trained tokenizer feeds the next stages: `tokenize_data` (binary from
`tokenizer/src/main.rs`, parquet/jsonl → `tokens.npy` + index files) and
`sample_tokenized.py` (epoch sampling; see its uint16 dtype note in
[gotchas.md](gotchas.md)).
