#!/usr/bin/env bash
# Render raw vs personalized tracking meshes for all completed sweep runs.
#
# Usage (repo root, WSL):
#   bash scripts/render_tracking_sweep.sh
#   ABLATION=default bash scripts/render_tracking_sweep.sh
#   RUN_NAME='justin*' SKIP_EXISTING=1 bash scripts/render_tracking_sweep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ABLATION="${ABLATION:-all}"
DEVICE="${DEVICE:-cuda}"
EXTRA=()

if [[ -n "${RUN_NAME:-}" ]]; then
  EXTRA+=(--run-name "$RUN_NAME")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA+=(--dry-run)
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA+=(--skip-existing)
fi
if [[ $# -gt 0 ]]; then
  EXTRA+=("$@")
fi

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "Repo:     $ROOT"
echo "Ablation: $ABLATION"
echo "Device:   $DEVICE"
echo ""

python scripts/render_tracking_sweep.py --ablation "$ABLATION" --device "$DEVICE" "${EXTRA[@]}"
