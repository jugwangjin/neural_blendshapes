#!/usr/bin/env bash
# Test-set PSNR/SSIM/LPIPS (FLARE masked) for completed sweep runs → Excel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ABLATION="${ABLATION:-all}"
DEVICE="${DEVICE:-cuda}"
EXTRA=()
[[ -n "${RUN_NAME:-}" ]] && EXTRA+=(--run-name "$RUN_NAME")
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA+=(--dry-run)
[[ "${SKIP_EXISTING:-0}" == "1" ]] && EXTRA+=(--skip-existing)
[[ $# -gt 0 ]] && EXTRA+=("$@")
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
python scripts/eval_test_image_metrics_sweep.py --ablation "$ABLATION" --device "$DEVICE" "${EXTRA[@]}"
