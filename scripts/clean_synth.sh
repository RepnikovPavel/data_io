#!/bin/bash
# Usage: ./scripts/clean_synth.sh [INPUT_DIR] [OUTPUT_DIR] [WORKERS]
set -e

INPUT_DIR="${1:-/mnt/hdd2/datasets_text/PleIAs/SYNTH}"
OUTPUT_DIR="${2:-/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/synth}"
WORKERS="${3:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

docker rm -f hrm_text_clean_synth 2>/dev/null || true
docker build -t hrm_text_clean_image -f "$PROJECT_ROOT/docker/DockerFileCleanStep" "$PROJECT_ROOT"

# Runs in foreground: logs stream live (python -u + PYTHONUNBUFFERED=1),
# container exits and is removed when the script finishes (Ctrl-C stops it).
docker run --rm \
  --name hrm_text_clean_synth \
  --user $(id -u):$(id -g) \
  -v "$PROJECT_ROOT":/workspace \
  -v "$INPUT_DIR":"$INPUT_DIR" \
  -v "$OUTPUT_DIR":"$OUTPUT_DIR" \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  hrm_text_clean_image \
  python -u pipe_clustered/clean_SYNTH.py \
    --input_dir "$INPUT_DIR" \
    --output_path "$OUTPUT_DIR" \
    --workers "$WORKERS"
