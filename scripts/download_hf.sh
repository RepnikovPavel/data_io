#!/bin/bash
# Download one HF dataset.
# Usage: ./scripts/download_hf.sh REPO_ID [DATASETS_DIR] [LOCAL_SUBDIR]
#   REPO_ID      e.g. openai/gsm8k
#   DATASETS_DIR default /mnt/hdd2/datasets_text
#   LOCAL_SUBDIR if set, files go to $DATASETS_DIR/$LOCAL_SUBDIR (plain layout),
#                otherwise into the HF cache ($DATASETS_DIR/.hf_cache) for load_dataset.
set -e

REPO_ID="$1"
DATASETS_DIR="${2:-/mnt/hdd2/datasets_text}"
LOCAL_SUBDIR="$3"

if [ -z "$REPO_ID" ]; then
  echo "Usage: $0 REPO_ID [DATASETS_DIR] [LOCAL_SUBDIR]" >&2
  exit 1
fi

NAME="$(echo "${REPO_ID}${LOCAL_SUBDIR:+/$LOCAL_SUBDIR}" | tr '/ ' '__')"
CONTAINER="hrm_text_download_${NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR"

docker rm -f "$CONTAINER" 2>/dev/null || true
docker build -t data_io_hrm_text_image -f "$PROJECT_ROOT/docker/DockerFileDownloadStep" "$PROJECT_ROOT"

# Foreground + --rm: logs stream live, container exits when download finishes.
if [ -n "$LOCAL_SUBDIR" ]; then
  HF_CMD="hf download $REPO_ID --repo-type dataset --local-dir $DATASETS_DIR/$LOCAL_SUBDIR --max-workers 8"
else
  HF_CMD="hf download $REPO_ID --repo-type dataset"
fi

docker run --rm \
  --name "$CONTAINER" \
  --user $(id -u):$(id -g) \
  -v "$DATASETS_DIR":"$DATASETS_DIR" \
  -w "$DATASETS_DIR" \
  -e HF_TOKEN=$HF_TOKEN \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e HF_HOME="$DATASETS_DIR/.hf_cache" \
  -e PYTHONUNBUFFERED=1 \
  data_io_hrm_text_image \
  bash -c "$HF_CMD && echo 'DONE: $REPO_ID'"
