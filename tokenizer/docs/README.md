# Tokenizer stage

Training the HRM-Text BPE tokenizer: byte-level BPE, vocab_size **65536**, NFC
normalizer, GPT-2-style pre-tokenizer regex, 31 special tokens (ids 0..30).

Subpages:

- [bpe_algorithm.md](bpe_algorithm.md) — BPE training in pseudocode for a
  reader unfamiliar with BPE: inputs, outputs, what happens inside
- [pipeline.md](pipeline.md) — how to run: download → clean → stage to NVMe →
  train; inputs/outputs; checkpoint & resume semantics of the iterative trainer
- [benchmarks.md](benchmarks.md) — measured timings/memory (48 cores, 251G RAM)
  and a cloud cost estimate
- [special_tokens.md](special_tokens.md) — all 31 added tokens: id, purpose in
  this project, origin
- [gotchas.md](gotchas.md) — known issues: the uint16 dtype bug, the
  tokenizers-crate i32 overflow, row-order sensitivity, non-TTY progress bars,
  container teardown
- [comparison.md](comparison.md) — authors' artifact vs baseline vs iterative
  vs C++: measured equality numbers

## Codebases

| dir | what |
|---|---|
| `tokenizer_orig/` | authors' original code, **frozen — never touch** |
| `tokenizer/` | our Rust crate: `train_tokenizer` (baseline = authors' algorithm + progress logging) and `train_tokenizer_iter` (iterative, checkpointable) |
| `tokenizer_cpp/` | C++17 port of the iterative trainer (for readers of C++): `train_tokenizer_cpp`, mode A from `words.bin`, mode B standalone from a text corpus (PCRE2) |

## Versions comparison (measured on this machine, full corpus)

Corpus: 5221 files, 410,012,296 docs, 170.2 GiB text → 12.1M unique words.

| | authors' (tokenizer_orig) | Rust baseline | Rust iterative | C++ port |
|---|---|---|---|---|
| Algorithm | tokenizers-crate BpeTrainer | same + progress logs | same, bug-for-bug | same, bug-for-bug |
| Load | 62.7 min (HDD) | 62.7 min (HDD) | ~30 min (NVMe) | — (reuses `words.bin`) |
| Train | 29.5 min | 29.5 min | 4.5 min → 0.7 min after the parity fix | 49 s |
| Total | ~92 min | ~92 min | ~35 min | load once + ~1 min |
| Peak memory | ~280G virt, ~220G swap spill | same | ~65G RAM (train phase alone: 7.0 GiB) | train phase: 4.0 GiB |
| Peak swap delta | ~220G | ~220G | 0 | 0 |
| Checkpointing | none | none | `words.bin` + `merges_N.bin`, SIGTERM-safe (exit 2), resume | same binary formats, cross-compatible with Rust |
| Purpose | reference behavior | reference run | production trainer | readability |

All three trainers produce the **same artifact**: iterative vs baseline vs C++
`tokenizer.json` are byte-identical ([proof](comparison.md)).

Artifacts land in `/mnt/hdd2/models/HRM-Text/tokenizers/{original,iterative,cpp}/bpe/`
(`tokenizer.json` + `config.json`).

## Quick run

```sh
./scripts/prepare_tokenizer_data.sh   # once: rsync HDD -> NVMe
./scripts/train_tokenizer_iter.sh     # Rust: load (~30 min, once) + train (~1 min)
./scripts/train_tokenizer_cpp.sh      # C++: from existing words.bin (~1 min)
./scripts/test_tokenizer_parity.sh /mnt/hdd2/models/HRM-Text/tokenizers/original/bpe/tokenizer.json \
    /mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe/tokenizer.json \
    /mnt/hdd2/models/HRM-Text/tokenizers/cpp/bpe/tokenizer.json
```

The trained tokenizer feeds the next stages: `tokenize_data` (binary from
`tokenizer/src/main.rs`, parquet/jsonl → `tokens.npy` + index files) and
`sample_tokenized.py` (epoch sampling; see its uint16 dtype note in
[gotchas.md](gotchas.md)).

## Building the C++ trainer

```sh
docker build -t hrm_text_tokenizer_cpp_image -f docker/DockerFileTokenizerCppStep .
# or locally:
cd tokenizer_cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# or plain g++ (PCRE2 optional, enables mode B):
g++ -O3 -std=c++17 -pthread src/*.cpp -o train_tokenizer_cpp \
    $(pkg-config --cflags --libs libpcre2-8) -DTOKENIZER_CPP_HAVE_PCRE2
```
