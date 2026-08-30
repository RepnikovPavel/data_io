# Comparison: authors' artifact vs baseline vs iterative

← [README.md](README.md)

Three tokenizer artifacts, all BPE vocab_size 65536:

- **authors'** — `trained_tokenizers/bpe/tokenizer.json` (ships with the repo;
  from the HRM-Text authors)
- **baseline** — `/mnt/hdd2/models/HRM-Text/tokenizers/original/bpe/tokenizer.json`
  (our run of `train_tokenizer` on the cleaned corpus)
- **iterative** — `/mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe/tokenizer.json`
  (our `train_tokenizer_iter` on the same corpus)
- **cpp** — `train_tokenizer_cpp` (`tokenizer_cpp/`, C++17 port) trained from
  the same `words.bin`; verified byte-identical to baseline/iterative

All numbers below are measured on the current files.

## Structure

Programmatically verified **identical** across all three: `normalizer` (NFC),
`pre_tokenizer` (Sequence[Split(regex, Isolated), ByteLevel{add_prefix_space:
false, trim_offsets: false, use_regex: false}]), `post_processor`, `decoder`
(ByteLevel, same params), `added_tokens` (all 31, same ids and flags), and
`model` fields other than vocab/merges (type BPE, no unk/dropout, fuse_unk
false, etc.). All three have 65264 merges and 65536 vocab entries; both
`config.json` files are `{"model_type": "qwen3"}`.

## Content equality

| comparison | merge common prefix | vocab token overlap | notes |
|---|---|---|---|
| baseline vs authors' | 216 / 65264 (equal positions overall: 4681) | 65406 / 65536 (99.80%) | divergence from row-order-sensitive sampling (see [gotchas.md](gotchas.md#row-order-sensitivity-of-per-file-sampling)) |
| **iterative vs baseline** | **65264 / 65264 (full)** | **65536 / 65536, incl. id assignment** | `tokenizer.json` and `config.json` are **byte-identical** (`cmp` clean) |
| **cpp vs baseline** | **65264 / 65264 (full)** | **65536 / 65536, incl. id assignment** | byte-identical; also byte-identical after a SIGTERM kill + resume mid-run |

("Merge common prefix" = first index where the merge lists differ; "equal
positions" = positions where the two lists agree, regardless of order. The
baseline↔authors' pair is prefix 216 / equal positions 4681 — an earlier
hand-quoted figure of "prefix 4681" conflated the two metrics.)

Segmentation equality was also measured directly with
`scripts/test_tokenizer_parity.py`: iterative vs baseline — 20/20 fixed test
strings and 2000/2000 sampled gsm8k_train records produce identical token
strings; authors' artifact — 20/20 fixed strings and 1998/2000 sampled records
(the 2 diffs reflect its 130-token vocab difference from the baseline).

## Why the iterative trainer matches exactly

It reproduces the reference algorithm bug-for-bug (see
[gotchas.md](gotchas.md#tokenizers-crate-i32-pair-count-overflow-present-in-both-tokenizers-on-purpose)):
corpus-derived initial alphabet (241 of 256 byte-level chars occur in this
corpus → 272 initial tokens), i32 wrapping pair counts, queue admission only
for positive-count / positive-change pairs, tie-break by ascending token-id
pair, min_frequency 2. A unit test
(`cargo test --release --bin train_tokenizer_iter`) trains both
implementations on the same word counts and asserts identical vocab and
merges; kill/resume during training is also covered (SIGTERM → checkpoint →
resume → byte-identical result).
