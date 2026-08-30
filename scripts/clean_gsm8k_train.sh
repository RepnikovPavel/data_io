#!/bin/bash
# Usage: ./scripts/clean_gsm8k_train.sh [DATASETS_DIR] [OUTPUT_DIR] [WORKERS]
set -e

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
OUTPUT_DIR="${2:-/mnt/hdd2/datasets_text_transformed/HRM-Text/data}"
WORKERS="${3:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR"
mkdir -p "$OUTPUT_DIR"

docker rm -f hrm_text_clean_gsm8k_train 2>/dev/null || true
docker build -t hrm_text_clean_image -f "$PROJECT_ROOT/docker/DockerFileCleanStep" "$PROJECT_ROOT"

# Runs in foreground: logs stream live (python -u + PYTHONUNBUFFERED=1),
# container exits and is removed when the script finishes (Ctrl-C stops it).
docker run --rm \
  --name hrm_text_clean_gsm8k_train \
  --user $(id -u):$(id -g) \
  -v "$PROJECT_ROOT":/workspace \
  -v "$DATASETS_DIR":"$DATASETS_DIR" \
  -v "$OUTPUT_DIR":"$OUTPUT_DIR" \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME="$DATASETS_DIR/.hf_cache" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_HUB_OFFLINE=1 \
  hrm_text_clean_image \
  python -u pipe/clean_gsm8k_train.py \
    --output_path "$OUTPUT_DIR/gsm8k_train.jsonl" \
    --workers "$WORKERS"
