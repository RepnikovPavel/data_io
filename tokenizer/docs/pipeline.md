# Pipeline procedure

← [README.md](README.md)

## Stages

```
raw download → clean (pipe/, pipe_clustered/) → stage to NVMe → train tokenizer
```

### 1. Download (already done)

```sh
nohup bash scripts/download_data.sh > download_data.txt 2>&1 &
```

See `docs/data_download.md`.

### 2. Clean (already done)

`scripts/clean_*.sh` wrap `pipe/*.py` / `pipe_clustered/*.py`. Output contract:
`{condition, instruction, response}` per row — JSONL under
`/mnt/hdd2/datasets_text_transformed/HRM-Text/data/`, parquet under
`.../data_clustered/`. See `scripts/docs/README.md` for per-dataset docs.

### 3. Stage to NVMe

```sh
./scripts/prepare_tokenizer_data.sh
# rsync -a /mnt/hdd2/datasets_text_transformed/HRM-Text/{data,data_clustered} $HOME/hrm_text_tokenizer_cache/
```

Args: `[TRANSFORMED_ROOT] [NVME_DIR]`. Reason: the load phase is I/O-heavy;
NVMe staging cut the iterative load from HDD-bound to ~30 min (see
[benchmarks.md](benchmarks.md)).

### 4a. Baseline train (authors' algorithm, `tokenizers`-crate BpeTrainer)

```sh
./scripts/train_tokenizer.sh
#   TRANSFORMED_ROOT=/mnt/hdd2/datasets_text_transformed/HRM-Text (reads from HDD)
#   OUT_DIR=/mnt/hdd2/models/HRM-Text/tokenizers/original/bpe
```

Wraps:

```sh
docker build -t hrm_text_tokenizer_image -f docker/DockerFileTokenizerStep .
docker run --rm --init -v $PWD:/workspace -v $TRANSFORMED_ROOT:$TRANSFORMED_ROOT -v $OUT_DIR:$OUT_DIR \
  hrm_text_tokenizer_image \
  train_tokenizer "$TRANSFORMED_ROOT/data" "$TRANSFORMED_ROOT/data_clustered" \
    -o "$OUT_DIR/tokenizer.json" --prefix-config /workspace/prefix_config.yaml
```

Loads ALL documents into RAM (instruction and response as separate documents,
each truncated to the first 10k chars), then trains in one shot. No
checkpointing: if it dies, it restarts from zero. Writes `tokenizer.json` +
`config.json` (see [gotchas.md](gotchas.md#why-configjson-says-model_type-qwen3)).

### 4b. Iterative train (ours, checkpointable)

```sh
./scripts/train_tokenizer_iter.sh
#   DATA_ROOT=$HOME/hrm_text_tokenizer_cache   (NVMe, from step 3)
#   OUT_DIR=/mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe
#   CHECKPOINT_DIR=$HOME/hrm_text_tokenizer_cache/_checkpoints
```

Wraps:

```sh
docker run --rm --init -v $PWD:/workspace -v $DATA_ROOT:$DATA_ROOT -v $OUT_DIR:$OUT_DIR \
  -v $CHECKPOINT_DIR:$CHECKPOINT_DIR \
  hrm_text_tokenizer_image \
  train_tokenizer_iter "$DATA_ROOT/data" "$DATA_ROOT/data_clustered" \
    -o "$OUT_DIR/tokenizer.json" --prefix-config /workspace/prefix_config.yaml \
    --checkpoint-dir "$CHECKPOINT_DIR"
```

CLI:

```
train_tokenizer_iter <DIRS>... -o <OUT_TOKENIZER_JSON> --prefix-config <PATH> --checkpoint-dir <DIR>
  [--seed 0] [--vocab-size 65536] [--tokenizer-type bpe]
  [--phase all|load|train]            # default all
  [--truncate-len 10000] [--limit-mul-factor 10]
  [--checkpoint-interval 100]
```

Same data pipeline as the baseline, bit for bit: per-doc 10k-char
(`truncate_safe`, char-boundary) truncation, per-file sampling from
`prefix_config.yaml` (`limit = limit_mul_factor * max_per_file`, deterministic
`partial_shuffle` with `Pcg64` seeded by `DefaultHasher(seed, safe_name)`),
NFC + the exact same split regex; word counts are merged per-thread, so they
do not depend on the thread count.

#### Checkpoint / resume semantics

- **Phase `load`** streams all files once (rayon over files), counts
  pre-tokenizer word frequencies, and writes
  `<checkpoint-dir>/words.bin` (magic `WBI1`; docs total, bytes total, then
  per unique word: `u32 len + bytes + u64 count`, sorted → deterministic).
  Never holds raw documents. If `words.bin` exists, the load phase is skipped
  (delete it to force a reload). Note: `words.bin` checkpoints only at the
  END of load — an interrupted load restarts from zero.
- **Phase `train`** loads `words.bin`, then runs the merge loop, writing
  `<checkpoint-dir>/merges_<N>.bin` every `--checkpoint-interval` merges
  (default 100). Format: magic `MBI1`, vocab so far (id order), merges so far.
- On **SIGINT/SIGTERM** the trainer writes `merges_<N>.bin` for the current N
  and exits with **code 2**. Re-running the same command resumes from the
  latest `merges_*.bin` (merges are replayed per word by rank, pair counts
  rebuilt from scratch — verified: kill + resume produces a byte-identical
  result vs an uninterrupted run).
- Only on reaching `vocab_size` (or exhausting pairs with count ≥
  min_frequency 2) does it write `tokenizer.json` + `config.json`.

#### Progress logging

indicatif progress bars are invisible in non-TTY logs (docker logs), so both
trainers print explicit lines: `[load]` every 15 s (files/docs/GiB/elapsed/ETA),
`[train]` heartbeat; the iterative trainer additionally prints every 25 merges
(merge index, pair, count, merges/s, ETA) and every checkpoint write.
