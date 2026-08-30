#!/bin/bash
# Copy transformed data to fast NVMe before tokenizer training.
# Usage: ./scripts/prepare_tokenizer_data.sh [TRANSFORMED_ROOT] [NVME_DIR]
set -e

TRANSFORMED_ROOT="${1:-/mnt/hdd2/datasets_text_transformed/HRM-Text}"
NVME_DIR="${2:-$HOME/hrm_text_tokenizer_cache}"

mkdir -p "$NVME_DIR"
rsync -a --info=progress2 "$TRANSFORMED_ROOT/data" "$TRANSFORMED_ROOT/data_clustered" "$NVME_DIR/"
echo "DONE: data staged to $NVME_DIR"
