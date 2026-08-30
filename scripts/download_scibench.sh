#!/bin/bash
# Clone scibench repo. Usage: ./scripts/download_scibench.sh [DATASETS_DIR]
set -e

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR/Platypus"

docker rm -f hrm_text_download_scibench 2>/dev/null || true
docker build -t data_io_hrm_text_image -f "$PROJECT_ROOT/docker/DockerFileDownloadStep" "$PROJECT_ROOT"

docker run --rm \
  --name hrm_text_download_scibench \
  --user $(id -u):$(id -g) \
  -v "$DATASETS_DIR":"$DATASETS_DIR" \
  -w "$DATASETS_DIR" \
  -e PYTHONUNBUFFERED=1 \
  data_io_hrm_text_image \
  bash -c "git clone https://github.com/mandyyyyii/scibench.git $DATASETS_DIR/Platypus/scibench \
           || (cd $DATASETS_DIR/Platypus/scibench && git pull); \
           echo 'DONE: scibench'"
