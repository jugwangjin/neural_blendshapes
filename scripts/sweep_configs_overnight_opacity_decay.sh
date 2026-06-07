#!/usr/bin/env bash
# Same as sweep_configs_overnight.sh but enables arxiv:2404.06109 Sec. 3.5 opacity regularization.
#
# - After each densification run: subtract 0.001 from every primitive opacity (no hard reset)
# - w_opacity_decay (head-region L1) is OFF
#
# Usage (repo root):
#   bash scripts/sweep_configs_overnight_opacity_decay.sh
#   GPUS="0 1" PATTERN="nbs_id*.txt" bash scripts/sweep_configs_overnight_opacity_decay.sh
#
# Logs: ``.../neural_blendshapes_10_opacity_decay/<run_name>``.
# Pass FORCE=1 to re-run completed subjects.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROUNDS="${ROUNDS:-2}"
SLEEP_SEC="${SLEEP_SEC:-1800}"
GPUS="${GPUS:-0}"
PATTERN="${PATTERN:-*.txt}"
OPACITY_DECAY_OVERRIDES="${OPACITY_DECAY_OVERRIDES:-configs/loss_overrides/opacity_decay.json}"
ABLATION="${ABLATION:-opacity_decay}"

EXTRA_ARGS=(--ablation "$ABLATION" --loss-overrides "$OPACITY_DECAY_OVERRIDES")

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
echo "Overrides:  $(basename "$OPACITY_DECAY_OVERRIDES")"
echo "Rounds:     $ROUNDS"
echo "Sleep:      ${SLEEP_SEC}s between rounds"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Log root:   /Bean/log/gwangjin/2026/neural_blendshapes_10_opacity_decay/<run_name>"
echo ""

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for ((i = 1; i <= ROUNDS; i++)); do
  echo "=============================="
  echo " sweep_configs (opacity_decay) round ${i}/${ROUNDS}  $(date -Is 2>/dev/null || date)"
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
