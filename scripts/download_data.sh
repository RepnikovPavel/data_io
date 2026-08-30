#!/bin/bash
# Download ALL datasets needed by the clean_* scripts.
# Usage: ./scripts/download_data.sh [DATASETS_DIR]
# Each dataset downloads via its own script (scripts/download_*.sh) with its own
# log (/tmp/hrm_text_download_<name>.log), so failures are isolated and progress
# is per-dataset. Re-run any single dataset with its script directly.
set -u

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$DATASETS_DIR/Open-Orca" "$DATASETS_DIR/PleIAs" "$DATASETS_DIR/Platypus"
find "$DATASETS_DIR/" -name '*.lock' -delete 2>/dev/null || true

# Pre-build the image once so parallel per-dataset scripts skip it.
docker build -t data_io_hrm_text_image -f "$PROJECT_ROOT/docker/DockerFileDownloadStep" "$PROJECT_ROOT"

HF_REPOS="
openai/gsm8k
EleutherAI/hendrycks_math
facebook/natural_reasoning
HuggingFaceH4/no_robots
AI-MO/NuminaMath-1.5
KbsdJames/Omni-MATH
facebook/principia-collection
TIGER-Lab/WebInstruct-verified
allenai/openbookqa
metaeval/reclor
metaeval/ScienceQA_text_only
TIGER-Lab/TheoremQA
nvidia/AceReason-1.1-SFT
nvidia/OpenMathInstruct-2
open-thoughts/OpenThoughts2-1M
sapientinc/sudoku-extreme
tasksource/tasksource-instruct-v0
MegaScience/TextbookReasoning
"

echo "== large datasets (plain layout) =="
"$SCRIPT_DIR/download_hf.sh" Open-Orca/FLAN "$DATASETS_DIR" Open-Orca/FLAN \
  > /tmp/hrm_text_download_flan.log 2>&1 &
"$SCRIPT_DIR/download_hf.sh" PleIAs/SYNTH "$DATASETS_DIR" PleIAs/SYNTH \
  > /tmp/hrm_text_download_synth.log 2>&1 &
"$SCRIPT_DIR/download_hf.sh" imone/ARB "$DATASETS_DIR" Platypus/ARB \
  > /tmp/hrm_text_download_arb.log 2>&1 &

echo "== misc =="
"$SCRIPT_DIR/download_amps.sh" "$DATASETS_DIR" \
  > /tmp/hrm_text_download_amps.log 2>&1 &
"$SCRIPT_DIR/download_scibench.sh" "$DATASETS_DIR" \
  > /tmp/hrm_text_download_scibench.log 2>&1 &
"$SCRIPT_DIR/download_math_dataset.sh" "$DATASETS_DIR" \
  > /tmp/hrm_text_download_math_dataset.log 2>&1 &

echo "== HF-cache datasets (used via load_dataset) =="
echo "$HF_REPOS" | xargs -P 6 -I{} bash -c '
  repo="{}"
  name=$(echo "$repo" | tr "/" "_")
  if "'"$SCRIPT_DIR"'/download_hf.sh" "$repo" "'"$DATASETS_DIR"'" > "/tmp/hrm_text_download_${name}.log" 2>&1; then
    echo "OK: $repo"
  else
    echo "FAILED: $repo (log: /tmp/hrm_text_download_${name}.log)"
  fi'

echo "== waiting for large/misc downloads =="
wait

echo "== summary =="
for log in /tmp/hrm_text_download_*.log; do
  if grep -q "^DONE:" "$log" 2>/dev/null; then
    echo "OK:     $(basename "$log" .log) ($(grep -c '^DONE:' "$log") done)"
  else
    echo "FAILED: $(basename "$log" .log)"
  fi
done
echo "All downloads complete!"
