# tokenizer_cpp — how the C++ trainer fits the pipeline

## The one thing to understand

**The C++ code never touches parquet/jsonl dataset files.** Reading
heterogeneous inputs (parquet, jsonl) is the *load phase*, and it stays in the
Rust trainer (`tokenizer/src/bin/train_tokenizer_iter.rs --phase load`), which
converts the whole corpus once into a single binary checkpoint:

```
transformed data (parquet/jsonl, 345 GiB, HDD/NVMe)
        │  ./scripts/train_tokenizer_iter.sh  --phase load   (Rust, ~30 min)
        ▼
words.bin  —  unique pretokenized words + counts
              (~12.1M words; the ONLY thing the merge loop needs)
        │
        ├─► Rust  train phase   (train_tokenizer_iter --phase train)
        └─► C++   train phase   (tokenizer_cpp --words words.bin)   ← this repo
```

BPE training is a pure function of **word → count** statistics. Once those
counts exist, the raw corpus is never needed again. So the C++ port implements
the part that matters for reading/understanding the algorithm — the merge
loop — against `words.bin`, instead of re-implementing parquet/arrow IO
(which in C++ means pulling in Apache Arrow C++; deliberately avoided).

`words.bin` format (little-endian, written atomically via tmp+rename):

```
magic "WBI1" | u64 docs_total | u64 bytes_total | u64 n_words |
n_words × { u32 word_len | word bytes (UTF-8, NFC, regex-pre-tokenized) | u64 count }
```

The merge checkpoints (`merges_N.bin`) are also byte-compatible both ways:
a run can be started in Rust, checkpointed, and **resumed in C++** (and vice
versa) with byte-identical output. This is exactly what
`scripts/train_tokenizer_cpp.sh` does by default: it points the C++ trainer at
the checkpoint dir produced by the Rust load phase.

## Running the C++ trainer

### A. From the official corpus (uses the existing words.bin)

```bash
# one-time: stage data to NVMe + build the word counts (Rust load phase)
./scripts/prepare_tokenizer_data.sh
./scripts/train_tokenizer_iter.sh   # or: ... --phase load equivalent

# then the C++ trainer consumes the checkpoint:
./scripts/train_tokenizer_cpp.sh ~/hrm_text_tokenizer_cache/_checkpoints \
    /mnt/hdd2/models/HRM-Text/tokenizers/cpp/bpe
```

Result: `tokenizer.json` + `config.json`, byte-identical to the Rust
iterative trainer's output (verified by `scripts/test_tokenizer_parity.sh`).

### B. Standalone (no Rust, small custom corpus) — for reading/experiments

```bash
# one document per line, UTF-8; NFC normalization is NOT applied in this mode
printf 'Hello world\nBPE is a greedy merge algorithm\n' > /tmp/corpus.txt
./scripts/train_tokenizer_cpp.sh   # builds the image
# or directly:
g++ -O3 -std=c++17 -pthread tokenizer_cpp/src/*.cpp \
    $(pkg-config --cflags --libs libpcre2-8) -DTOKENIZER_CPP_HAVE_PCRE2 \
    -o /tmp/train_cpp
/tmp/train_cpp --corpus /tmp/corpus.txt -o /tmp/out/tokenizer.json \
    --vocab-size 1000 --checkpoint-dir /tmp/out/ckpt
```

Mode B exists so the whole algorithm (regex pre-tokenization → byte-level
alphabet → word counts → merge loop → tokenizer.json) is readable in one
C++ file set. Differences from mode A: corpus text lines are pretokenized
with PCRE2 (`\p{L}` Unicode classes), NFC is skipped (documented in
`pretok.h`), and no `prefix_config.yaml` sampling limits apply.

## Source map

| file | contents |
|---|---|
| `src/main.cpp` | CLI (`--words` / `--corpus`, `-o`, `--vocab-size`, `--checkpoint-dir`, `--threads`) |
| `src/pretok.*` | mode B: PCRE2 regex split + word counting (needs libpcre2-8) |
| `src/byte_level.h` | GPT-2 bytes↔unicode table (bijection), UTF-8 encode, log rendering |
| `src/checkpoint.*` | `words.bin` / `merges_N.bin` IO (cross-compatible with Rust) |
| `src/trainer.*` | the BPE merge loop (incremental pair counts, lazy max-heap, checkpoints, SIGTERM-safe exit 2) |
| `src/json_writer.*` | HF `tokenizer.json` writer (serde_json-pretty compatible) + `config.json` |

Build: `docker build -t hrm_text_tokenizer_cpp_image -f docker/DockerFileTokenizerCppStep .`
or CMake (`tokenizer_cpp/CMakeLists.txt`).

## See also

- `tokenizer/docs/README.md` — index of the whole tokenizer stage (three
  codebases: `tokenizer_orig/` authors' frozen, `tokenizer/` our Rust,
  `tokenizer_cpp/` this port)
- `tokenizer/docs/benchmarks.md` — measured timings/RAM for all three
- `tokenizer/docs/comparison.md` — parity results vs the authors' artifact
