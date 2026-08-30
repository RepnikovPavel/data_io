#!/bin/bash
# Download DeepMind mathematics_dataset. Usage: ./scripts/download_math_dataset.sh [DATASETS_DIR]
set -e

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR"

docker rm -f hrm_text_download_math_dataset 2>/dev/null || true
docker build -t data_io_hrm_text_image -f "$PROJECT_ROOT/docker/DockerFileDownloadStep" "$PROJECT_ROOT"

docker run --rm \
  --name hrm_text_download_math_dataset \
  --user $(id -u):$(id -g) \
  -v "$DATASETS_DIR":"$DATASETS_DIR" \
  -w "$DATASETS_DIR" \
  -e PYTHONUNBUFFERED=1 \
  data_io_hrm_text_image \
  bash -c "wget -O $DATASETS_DIR/mathematics_dataset-v1.0.tar.gz 'https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz' \
           && tar -xzvf $DATASETS_DIR/mathematics_dataset-v1.0.tar.gz -C $DATASETS_DIR \
           && rm $DATASETS_DIR/mathematics_dataset-v1.0.tar.gz \
           && echo 'DONE: mathematics_dataset'"
