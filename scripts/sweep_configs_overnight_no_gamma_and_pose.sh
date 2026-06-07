#!/usr/bin/env bash
# Same as sweep_configs_overnight.sh but every train.py uses the no-gamma-and-pose schedule.
#
# Tracker: no gamma, no pose (raw ICT coeffs); template + expr_mlp + GS from stage 2.
#
# Usage (repo root):
#   bash scripts/sweep_configs_overnight_no_gamma_and_pose.sh
#   GPUS="0 1" PATTERN="nbs_id*.txt" bash scripts/sweep_configs_overnight_no_gamma_and_pose.sh
#
# Logs: ``.../neural_blendshapes_10_no_gamma_and_pose/<run_name>``.
# Pass FORCE=1 to re-run completed subjects.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROUNDS="${ROUNDS:-2}"
SLEEP_SEC="${SLEEP_SEC:-1800}"
GPUS="${GPUS:-3}"
PATTERN="${PATTERN:-*.txt}"
NO_GP_OVERRIDES="${NO_GP_OVERRIDES:-configs/loss_overrides/no_gamma_and_pose.json}"
ABLATION="${ABLATION:-no_gamma_and_pose}"

EXTRA_ARGS=(--ablation "$ABLATION" --loss-overrides "$NO_GP_OVERRIDES")

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
echo "Schedule:   no_gamma_and_pose ($(basename "$NO_GP_OVERRIDES"))"
echo "Rounds:     $ROUNDS"
echo "Sleep:      ${SLEEP_SEC}s between rounds"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Log root:   /Bean/log/gwangjin/2026/neural_blendshapes_10_no_gamma_and_pose/<run_name>"
echo ""

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for ((i = 1; i <= ROUNDS; i++)); do
  echo "=============================="
  echo " sweep_configs (no_gamma_and_pose) round ${i}/${ROUNDS}  $(date -Is 2>/dev/null || date)"
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
