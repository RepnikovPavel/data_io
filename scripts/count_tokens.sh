#!/bin/bash
# Count tokens over the ENTIRE transformed corpus with the trained tokenizer.
# Usage: ./scripts/count_tokens.sh [CORPUS_ROOT] [OUT_DIR]
#   CORPUS_ROOT default ~/hrm_text_tokenizer_cache (NVMe copy of the transformed corpus)
#   OUT_DIR     default scripts/docs (token_counts.json + token_counts.md land here)
#
# Steps: 1) stage corpus -> .tokbin mirror (clean image, skips up-to-date files)
#        2) C++ count_tokens over all .tokbin (tokenizer_cpp image)
#        3) aggregate TSV -> scripts/docs/token_counts.{json,md} (clean image)
set -e

CORPUS_ROOT="${1:-$HOME/hrm_text_tokenizer_cache}"
OUT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docs}"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TOKBIN_DIR="${TOKBIN_DIR:-$HOME/hrm_text_tokbin}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/mnt/hdd2/models/HRM-Text/tokenizers/iterative/bpe/tokenizer.json}"
TSV="$TOKBIN_DIR/counts.tsv"

mkdir -p "$TOKBIN_DIR"

echo "=== [1/3] staging corpus -> .tokbin ($TOKBIN_DIR)"
docker run --rm --init \
  --name hrm_text_count_tokens_stage \
  --user $(id -u):$(id -g) \
  -e PYTHONUNBUFFERED=1 \
  -v "$PROJECT_ROOT":/workspace \
  -v "$CORPUS_ROOT":"$CORPUS_ROOT" \
  -v "$TOKBIN_DIR":"$TOKBIN_DIR" \
  -w /workspace \
  hrm_text_clean_image \
  python3 scripts/stage_corpus_tokbin.py "$CORPUS_ROOT" "$TOKBIN_DIR"

echo "=== [2/3] counting tokens (C++)"
docker rm -f hrm_text_count_tokens 2>/dev/null || true
docker build -t hrm_text_tokenizer_cpp_image -f "$PROJECT_ROOT/docker/DockerFileTokenizerCppStep" "$PROJECT_ROOT"
docker run --rm --init \
  --name hrm_text_count_tokens \
  --user $(id -u):$(id -g) \
  -v "$TOKBIN_DIR":"$TOKBIN_DIR" \
  -v "$(dirname "$TOKENIZER_JSON")":"$(dirname "$TOKENIZER_JSON")" \
  -w /workspace \
  hrm_text_tokenizer_cpp_image \
  count_tokens --tokenizer "$TOKENIZER_JSON" -o "$TSV" "$TOKBIN_DIR"

echo "=== [3/3] aggregate -> $OUT_DIR/token_counts.{json,md}"
docker run --rm --init \
  --user $(id -u):$(id -g) \
  -v "$PROJECT_ROOT":/workspace \
  -v "$TOKBIN_DIR":"$TOKBIN_DIR" \
  -v "$OUT_DIR":"$OUT_DIR" \
  -w /workspace \
  hrm_text_clean_image \
  python3 scripts/aggregate_token_counts.py "$TSV" "$OUT_DIR" "$TOKENIZER_JSON"

echo "DONE: $OUT_DIR/token_counts.json + token_counts.md"
