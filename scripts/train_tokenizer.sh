#!/bin/bash
# Usage: ./scripts/train_tokenizer.sh [TRANSFORMED_ROOT] [OUT_DIR]
set -e

TRANSFORMED_ROOT="${1:-/mnt/hdd2/datasets_text_transformed/HRM-Text}"
OUT_DIR="${2:-/mnt/hdd2/models/HRM-Text/tokenizers/original/bpe}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUT_DIR"

docker rm -f hrm_text_train_tokenizer 2>/dev/null || true
docker build -t hrm_text_tokenizer_image -f "$PROJECT_ROOT/docker/DockerFileTokenizerStep" "$PROJECT_ROOT"

# Foreground + --rm: progress bars stream live, container exits when done.
# stderr (indicatif progress) is unbuffered by design.
docker run --rm \
  --name hrm_text_train_tokenizer \
  --user $(id -u):$(id -g) \
  -v "$PROJECT_ROOT":/workspace \
  -v "$TRANSFORMED_ROOT":"$TRANSFORMED_ROOT" \
  -v "$OUT_DIR":"$OUT_DIR" \
  -w /workspace \
  hrm_text_tokenizer_image \
  train_tokenizer \
    "$TRANSFORMED_ROOT/data" "$TRANSFORMED_ROOT/data_clustered" \
    -o "$OUT_DIR/tokenizer.json" \
    --prefix-config /workspace/prefix_config.yaml
