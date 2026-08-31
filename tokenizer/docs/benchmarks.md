# Benchmarks

← [README.md](README.md)

Measured on this machine: 48 cores, 251G RAM, HDD `/mnt/hdd2` + NVMe.
Corpus: 5221 files, 410,012,296 documents (instruction and response counted
separately), 170.2 GiB text, 12.1M unique pre-tokenizer words.

| | Baseline (`train_tokenizer`, data on HDD) | Iterative (`train_tokenizer_iter`, data on NVMe) | C++ port (`train_tokenizer_cpp`, from `words.bin`) |
|---|---|---|---|
| Load | 62.7 min | ~30 min* | — (reuses the Rust `words.bin`) |
| Train | 29.5 min | 4.5 min** | 49 s |
| **Total** | **~92 min** | **~35 min** | load once + ~1 min |
| Peak memory | ~280G virtual — spilled ~220G into NVMe swap; RAM-only would need ~300G or OOM | ~65G RAM, no swap (train phase alone: 7.0 GiB, 0 swap) | train phase: 4.0 GiB, 0 swap |
| Checkpointing | none | `words.bin` + `merges_N.bin` every 100 merges, SIGTERM-safe | same formats, cross-compatible with Rust |
| Result | baseline | byte-identical to baseline | byte-identical to baseline |

Train-phase-only numbers measured with `/tmp/bench_train_monitor.sh` (docker
stats every 5 s, includes page cache): Rust iterative 49 s wall / 7.0 GiB peak
/ 0 swap; C++ 49 s wall / 4.0 GiB peak / 0 swap. Both started from the same
`words.bin` and produced byte-identical output to the official
`iterative/bpe/tokenizer.json` (`cmp` clean), and a SIGTERM kill + resume
mid-run also produced byte-identical output for both.

\* Includes a ~15 min single-threaded tail on the last huge file — the load
parallelizes across files, not within one file. Known headroom: intra-file
parallelism would shave most of the tail.

\** Measured with the first parity version; the final version (per-site
incremental updates) runs the merge loop in ~1 min. Load dominates.

Why the memory difference: the baseline keeps all 410M documents (170 GiB of
Rust `String`s, plus per-doc overhead and the trainer's own structures) in RAM
for the whole run. The iterative trainer never holds raw text: `words.bin` is
~290 MB (12.1M unique words), and the train phase works on token-id sequences
of unique words only.

## Logging gotcha

indicatif progress bars are invisible in non-TTY output (`docker logs`). Both
trainers therefore emit explicit progress lines: `[load]` every 15 s,
`[train]` heartbeat every 60 s; the iterative trainer also prints every 25
merges with rate and ETA.

## Cloud cost estimate

Formula: `cost ≈ instance_price_per_hour × wall_hours` (storage negligible;
~400G disk for corpus + checkpoints).

| Trainer | Instance (example, on-demand us-east-1) | Why | Time | Cost |
|---|---|---|---|---|
| Iterative | i4i.8xlarge — 32 vCPU / 256G / NVMe, ~$2.7/hr | NVMe for load; 256G ≥ ~70G needed | ~40 min | **≈ $1.8** |
| Baseline | r7i.16xlarge — 64 vCPU / 512G, ~$4.3/hr | needs ≥300G RAM to avoid swap | ~1.5 h | **≈ $6.5** |

CPU/RAM requirements, stated explicitly:

- **Load phase** parallelizes across files — 32–48 cores recommended; scales
  near-linearly until I/O-bound (hence NVMe).
- **Merge loop** is single-heavy (one core mostly) and memory-bound; extra
  cores only help the parallel pair-count init and word replay on resume.
- RAM: **≥ 70G** for the iterative trainer, **≥ 300G** for the baseline (or
  accept ~220G of swap and the slowdown).

## Token counting stage (C++ `count_tokens` over the full corpus)

Corpus-wide token counting with the trained tokenizer
(`scripts/count_tokens.sh`: parquet/jsonl → .tokbin staging → C++ encode+count
→ `scripts/docs/token_counts.json`). Measured 2026-09-01 on the full corpus
(594,025,764 rows → 1,188,051,528 docs, 632 GiB text):

| stage | time | rate | peak RAM | notes |
|---|---|---|---|---|
| staging → .tokbin | ~2.7 min (warm) / ~4 min (cold flan) | ~2.6 GiB/s | bounded (streaming) | 626 GiB intermediate on NVMe |
| count (48 threads) | **31.5 min** | 0.33 GiB/s ≈ **93.2M tokens/s** | 170 GiB (≈all mmap page cache; heap is a few GiB) | uint16 token ids |
| aggregate | seconds | — | — | |

Total wall: **~34 min**. Result: **176,125,636,509 tokens**
(`scripts/docs/token_counts.md`, per-dataset table + TOTAL row; the paper's
"40B unique tokens" is the sampled training mix, this count is the raw
unsampled corpus).

Encoder throughput (`count_tokens --bench`, tokens/s, 1-thread / 48-thread):
~100-char docs: 4.8M / 17.0M; ~1k chars: 6.6M / 54.1M; ~10k chars: 6.2M /
68.6M. Optimization history on a 13 GiB probe: 3.8 min (PCRE2 interpreter) →
1.3 min (PCRE2 JIT) → 0.75 min (hand-rolled scanner + mmap + uint16).
Validated per-doc against python `tokenizers` 0.23.1 (exact match).

Cloud cost for this stage: CPU-bound, so core count drives time:
`cost ≈ price/hr × 31.5 min × (48 / cores)`. Examples: c7i.12xlarge (48 vCPU,
~$2.14/hr) ≈ **$1.1**; c7i.4xlarge (16 vCPU, ~$0.71/hr) ≈ **$1.1** in ~1.6 h.
Disk: ~650G NVMe scratch for .tokbin (instance storage, e.g. i4i/c7i+d, or
stream without staging to skip it). RAM: any (page cache only; cap with
`--memory` if needed).
