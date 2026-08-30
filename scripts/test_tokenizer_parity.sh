#!/bin/bash
# Tokenizer parity check: candidates vs reference (structure, vocab, merges, segmentation).
# Usage: ./scripts/test_tokenizer_parity.py via docker:
#   ./scripts/test_tokenizer_parity.sh <reference.json> <candidate.json> [candidate2.json ...]
# Example:
#   ./scripts/test_tokenizer_parity.sh \
#     /mnt/hdd2/models/HRM-Text/tokenizers/original/bpe/tokenizer.json \
#     /mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe/tokenizer.json \
#     /mnt/hdd2/models/HRM-Text/tokenizers/cpp/bpe/tokenizer.json
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker build -t hrm_text_clean_image -f "$PROJECT_ROOT/docker/DockerFileCleanStep" "$PROJECT_ROOT"

docker run --rm \
  -v "$PROJECT_ROOT":/workspace \
  -v /mnt/hdd2/models:/mnt/hdd2/models \
  -v /mnt/hdd2/datasets_text_transformed:/mnt/hdd2/datasets_text_transformed \
  -w /workspace \
  hrm_text_clean_image \
  python3 /workspace/scripts/test_tokenizer_parity.py "$@"
