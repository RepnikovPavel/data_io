#!/bin/bash
# Download AMPS (original Google Drive link is dead; HF mirror minimalt/MATH_amps).
# Usage: ./scripts/download_amps.sh [DATASETS_DIR]
set -e

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR"

docker rm -f hrm_text_download_amps 2>/dev/null || true
docker build -t data_io_hrm_text_image -f "$PROJECT_ROOT/docker/DockerFileDownloadStep" "$PROJECT_ROOT"

docker run --rm \
  --name hrm_text_download_amps \
  --user $(id -u):$(id -g) \
  -v "$DATASETS_DIR":"$DATASETS_DIR" \
  -w "$DATASETS_DIR" \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e HF_HOME="$DATASETS_DIR/.hf_cache" \
  -e PYTHONUNBUFFERED=1 \
  data_io_hrm_text_image \
  bash -c "hf download minimalt/MATH_amps amps.tar.gz --repo-type dataset --local-dir $DATASETS_DIR \
           && tar -xzf $DATASETS_DIR/amps.tar.gz -C $DATASETS_DIR \
           && echo 'DONE: amps'"
