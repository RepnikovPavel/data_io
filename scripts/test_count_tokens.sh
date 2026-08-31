#!/bin/bash
# Self-test: C++ count_tokens encoder vs python `tokenizers` lib, per doc.
# Usage: ./scripts/test_count_tokens.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORK=/tmp/count_tokens_selftest
TOKENIZER_JSON=/mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe/tokenizer.json

rm -rf "$WORK" && mkdir -p "$WORK"

# 1) sample 1000 gsm8k rows (+ edge cases) -> docs.tokbin (clean image)
docker run --rm \
  -v "$PROJECT_ROOT":/workspace \
  -v /mnt/hdd2/models:/mnt/hdd2/models \
  -v /mnt/hdd2/datasets_text_transformed:/mnt/hdd2/datasets_text_transformed \
  -v "$WORK":"$WORK" \
  -w /workspace \
  hrm_text_clean_image \
  python3 scripts/test_count_tokens.py --write-tokbin "$WORK/docs.tokbin"

# 2) C++ per-doc counts (cpp image)
docker run --rm \
  -v "$WORK":"$WORK" \
  -v /mnt/hdd2/models:/mnt/hdd2/models \
  hrm_text_tokenizer_cpp_image \
  count_tokens --tokenizer "$TOKENIZER_JSON" --per-doc "$WORK/docs.tokbin" \
  > "$WORK/cpp_counts.txt"

# 3) python tokenizers per-doc counts + compare (clean image)
docker run --rm \
  -v "$PROJECT_ROOT":/workspace \
  -v /mnt/hdd2/models:/mnt/hdd2/models \
  -v "$WORK":"$WORK" \
  -w /workspace \
  hrm_text_clean_image \
  python3 scripts/test_count_tokens.py --check "$WORK/docs.tokbin" "$WORK/cpp_counts.txt"
