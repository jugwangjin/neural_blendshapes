#!/usr/bin/env bash
# Same as sweep_configs_overnight.sh but ICT coeffs use additive gamma residual (not pow).

# Usage (repo root):
#   bash scripts/sweep_configs_overnight_additive_gamma.sh
#   GPUS="0 1" PATTERN="nbs_id*.txt" bash scripts/sweep_configs_overnight_additive_gamma.sh
#
# Logs: ``.../neural_blendshapes_10_additive_gamma/<run_name>``.
# Pass FORCE=1 to re-run completed subjects.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROUNDS="${ROUNDS:-2}"
SLEEP_SEC="${SLEEP_SEC:-1800}"
GPUS="${GPUS:-2}"
PATTERN="${PATTERN:-*.txt}"
ADD_GAMMA_OVERRIDES="${ADD_GAMMA_OVERRIDES:-configs/loss_overrides/additive_gamma.json}"
ABLATION="${ABLATION:-additive_gamma}"

EXTRA_ARGS=(--ablation "$ABLATION" --loss-overrides "$ADD_GAMMA_OVERRIDES")

if [[ -n "${PATTERN:-}" ]]; then
  EXTRA_ARGS+=(--pattern "$PATTERN")
fi

if [[ -n "${GPUS:-}" ]]; then
  # shellcheck disable=SC2206
  GPU_ARR=($GPUS)
  EXTRA_ARGS+=(--gpus "${GPU_ARR[@]}")
fi

if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

if [[ $# -gt 0 ]]; then
  EXTRA_ARGS+=("$@")
fi

echo "Repo:       $ROOT"
echo "Ablation:   $ABLATION"
echo "Schedule:   additive_gamma ($(basename "$ADD_GAMMA_OVERRIDES"))"
echo "Rounds:     $ROUNDS"
echo "Sleep:      ${SLEEP_SEC}s between rounds"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Log root:   /Bean/log/gwangjin/2026/neural_blendshapes_10_additive_gamma/<run_name>"
echo ""

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for ((i = 1; i <= ROUNDS; i++)); do
  echo "=============================="
  echo " sweep_configs (additive_gamma) round ${i}/${ROUNDS}  $(date -Is 2>/dev/null || date)"
  echo "=============================="
  python sweep_configs.py "${EXTRA_ARGS[@]}" || true

  if (( i < ROUNDS )); then
    echo ""
    echo "Sleep ${SLEEP_SEC}s before next round..."
    sleep "$SLEEP_SEC"
    echo ""
  fi
done

echo "All ${ROUNDS} round(s) finished."
