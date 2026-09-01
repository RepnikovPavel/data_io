#!/bin/bash
# Render the per-domain token bar plot from scripts/docs/token_counts.json.
# Usage: ./scripts/plot_domain_tokens.sh [extra plot_domain_tokens.py args]
# Output: assets/domain_token_distribution.png (override: pass an out path arg).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker rm -f hrm_text_plot_domain_tokens 2>/dev/null || true
docker build -t hrm_text_plot_image -f "$PROJECT_ROOT/docker/DockerFilePlotStep" "$PROJECT_ROOT"

docker run --rm --init \
  --name hrm_text_plot_domain_tokens \
  --user $(id -u):$(id -g) \
  -v "$PROJECT_ROOT":/workspace \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  hrm_text_plot_image \
  python3 scripts/plot_domain_tokens.py "$@"

echo "DONE: assets/domain_token_distribution.png"
