#!/bin/bash
# Usage: ./scripts/clean_openmathinstruct2.sh [DATASETS_DIR] [OUTPUT_DIR] [WORKERS]
set -e

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
OUTPUT_DIR="${2:-/mnt/hdd2/datasets_text_transformed/HRM-Text/data_clustered/openmathinstruct2}"
WORKERS="${3:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR"
mkdir -p "$OUTPUT_DIR"

docker rm -f hrm_text_clean_openmathinstruct2 2>/dev/null || true
docker build -t hrm_text_clean_image -f "$PROJECT_ROOT/docker/DockerFileCleanStep" "$PROJECT_ROOT"

# Runs in foreground: logs stream live (python -u + PYTHONUNBUFFERED=1),
# container exits and is removed when the script finishes (Ctrl-C stops it).
# DATASETS_DIR is mounted 1:1 and doubles as the HF cache (HF_HOME), so the
# ~30GB nvidia/OpenMathInstruct-2 download is reused across runs.
docker run --rm \
  --name hrm_text_clean_openmathinstruct2 \
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
  python -u pipe_clustered/clean_openmathinstruct2.py \
    --output_path "$OUTPUT_DIR" \
    --workers "$WORKERS"
