#!/bin/bash
# C++ BPE trainer (tokenizer_cpp) from an existing words.bin checkpoint.
# Usage: ./scripts/train_tokenizer_cpp.sh [CHECKPOINT_DIR] [OUT_DIR]
set -e

CHECKPOINT_DIR="${1:-$HOME/hrm_text_tokenizer_cache/_checkpoints}"
OUT_DIR="${2:-/mnt/hdd2/models/HRM-Text/tokenizers/cpp/bpe}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUT_DIR" "$CHECKPOINT_DIR"

docker rm -f hrm_text_train_tokenizer_cpp 2>/dev/null || true
docker build -t hrm_text_tokenizer_cpp_image -f "$PROJECT_ROOT/docker/DockerFileTokenizerCppStep" "$PROJECT_ROOT"

# Foreground + --rm + --init: progress lines stream live, SIGTERM checkpoints and exits.
docker run --rm --init \
  --name hrm_text_train_tokenizer_cpp \
  --user $(id -u):$(id -g) \
  -v "$CHECKPOINT_DIR":"$CHECKPOINT_DIR" \
  -v "$OUT_DIR":"$OUT_DIR" \
  -w /workspace \
  hrm_text_tokenizer_cpp_image \
  train_tokenizer_cpp \
    --words "$CHECKPOINT_DIR/words.bin" \
    -o "$OUT_DIR/tokenizer.json" \
    --checkpoint-dir "$CHECKPOINT_DIR"
