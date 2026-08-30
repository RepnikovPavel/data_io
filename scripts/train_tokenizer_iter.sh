#!/bin/bash
# Iterative, checkpointable BPE training (resumable, per-merge progress).
# Usage: ./scripts/train_tokenizer_iter.sh [DATA_ROOT] [OUT_DIR] [CHECKPOINT_DIR]
set -e

DATA_ROOT="${1:-$HOME/hrm_text_tokenizer_cache}"
OUT_DIR="${2:-/mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe}"
CHECKPOINT_DIR="${3:-$HOME/hrm_text_tokenizer_cache/_checkpoints}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUT_DIR" "$CHECKPOINT_DIR"

docker rm -f hrm_text_train_tokenizer_iter 2>/dev/null || true
docker build -t hrm_text_tokenizer_image -f "$PROJECT_ROOT/docker/DockerFileTokenizerStep" "$PROJECT_ROOT"

# Foreground + --rm + --init: progress lines stream live, SIGTERM checkpoints and exits.
docker run --rm --init \
  --name hrm_text_train_tokenizer_iter \
  --user $(id -u):$(id -g) \
  -v "$PROJECT_ROOT":/workspace \
  -v "$DATA_ROOT":"$DATA_ROOT" \
  -v "$OUT_DIR":"$OUT_DIR" \
  -v "$CHECKPOINT_DIR":"$CHECKPOINT_DIR" \
  -w /workspace \
  hrm_text_tokenizer_image \
  train_tokenizer_iter \
    "$DATA_ROOT/data" "$DATA_ROOT/data_clustered" \
    -o "$OUT_DIR/tokenizer.json" \
    --prefix-config /workspace/prefix_config.yaml \
    --checkpoint-dir "$CHECKPOINT_DIR"
