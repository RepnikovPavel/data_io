#!/bin/bash
# Run ALL clean_* transforms sequentially (one HDD: parallel runs only thrash it).
# Usage: ./scripts/run_clean_all.sh [DATASETS_DIR] [OUT_ROOT]
# Per-job log: /tmp/hrm_text_clean_<name>.log. Failures don't stop the queue;
# re-run a single job with its scripts/clean_<name>.sh directly.
set -u

DATASETS_DIR="${1:-/mnt/hdd2/datasets_text}"
OUT_ROOT="${2:-/mnt/hdd2/datasets_text_transformed/HRM-Text}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {  # name, script, args...
  local name=$1 script=$2; shift 2
  echo "=== $(date +%H:%M:%S) start: $name"
  if "$SCRIPT_DIR/$script" "$@" > "/tmp/hrm_text_clean_${name}.log" 2>&1; then
    echo "=== $(date +%H:%M:%S) OK: $name"
  else
    echo "=== $(date +%H:%M:%S) FAILED: $name (log: /tmp/hrm_text_clean_${name}.log)"
  fi
}

# local-data jobs first (no download dependency)
run dmmath               clean_dmmath.sh               "$DATASETS_DIR/mathematics_dataset-v1.0" "$OUT_ROOT/data_clustered/dmmath"
run arb                  clean_arb.sh                  "$DATASETS_DIR/Platypus/ARB" "$OUT_ROOT/data/Platypus"
run scibench             clean_scibench.sh             "$DATASETS_DIR/Platypus/scibench/dataset/original" "$OUT_ROOT/data/Platypus"
run amps_khan            clean_amps_khan.sh            "$DATASETS_DIR/amps/khan" "$OUT_ROOT/data"
run ampsmathematica      clean_ampsmathematica.sh      "$DATASETS_DIR/amps.tar.gz" "$OUT_ROOT/data_clustered/ampsmathematica"

# HF-cache jobs (need download stage finished)
run gsm8k_train          clean_gsm8k_train.sh          "$DATASETS_DIR" "$OUT_ROOT/data"
run math_train           clean_math_train.sh           "$DATASETS_DIR" "$OUT_ROOT/data"
run natural_reasoning    clean_natural_reasoning.sh    "$DATASETS_DIR" "$OUT_ROOT/data"
run no_robots            clean_no_robots.sh            "$DATASETS_DIR" "$OUT_ROOT/data"
run numinamath           clean_numinamath.sh           "$DATASETS_DIR" "$OUT_ROOT/data"
run omnimath             clean_omnimath.sh             "$DATASETS_DIR" "$OUT_ROOT/data"
run principia_collection clean_principia_collection.sh "$DATASETS_DIR" "$OUT_ROOT/data"
run webinstruct_verified clean_webinstruct_verified.sh "$DATASETS_DIR" "$OUT_ROOT/data"
run openbookqa           clean_openbookqa.sh           "$DATASETS_DIR" "$OUT_ROOT/data/Platypus"
run reclor               clean_reclor.sh               "$DATASETS_DIR" "$OUT_ROOT/data/Platypus"
run scienceqa            clean_scienceqa.sh            "$DATASETS_DIR" "$OUT_ROOT/data/Platypus"
run theoremqa            clean_theoremqa.sh            "$DATASETS_DIR" "$OUT_ROOT/data/Platypus"
run acereason            clean_acereason.sh            "$DATASETS_DIR" "$OUT_ROOT/data_clustered/acereason"
run openthoughts2        clean_openthoughts2.sh        "$DATASETS_DIR" "$OUT_ROOT/data_clustered/openthoughts2"
run sudoku               clean_sudoku.sh               "$DATASETS_DIR" "$OUT_ROOT/data_clustered/sudoku_extreme"
run tasksource           clean_tasksource.sh           "$DATASETS_DIR" "$OUT_ROOT/data_clustered/tasksource"
run textbookreasoning    clean_textbookreasoning.sh    "$DATASETS_DIR" "$OUT_ROOT/data_clustered/textbookreasoning"
run openmathinstruct2    clean_openmathinstruct2.sh    "$DATASETS_DIR" "$OUT_ROOT/data_clustered/openmathinstruct2"

echo "=== $(date +%H:%M:%S) queue finished"
