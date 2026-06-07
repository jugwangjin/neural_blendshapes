#!/usr/bin/env bash
# Run sweep_configs.py repeatedly while NeRFBlendShape preprocessing catches up.
#
# Usage (repo root):
#   bash scripts/sweep_configs_overnight.sh
#   GPUS="0 1" PATTERN="nbs_id*.txt" bash scripts/sweep_configs_overnight.sh
# Multi-GPU default (0 1 2 3): scripts/sweep_configs_overnight_multigpu.sh
#   ROUNDS=5 SLEEP_SEC=3600 bash scripts/sweep_configs_overnight.sh
#   REVERSE=1 bash scripts/sweep_configs_overnight.sh
#   SHUFFLE=1 bash scripts/sweep_configs_overnight.sh
#   SHUFFLE=1 SHUFFLE_SEED=42 bash scripts/sweep_configs_overnight.sh
#
# Completed runs (stage_3_expression_detail_end_step_*.pt) are skipped by default.
# Pass FORCE=1 to sweep_configs via: FORCE=1 bash scripts/sweep_configs_overnight.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROUNDS="${ROUNDS:-2}"
SLEEP_SEC="${SLEEP_SEC:-1800}"
GPUS="${GPUS:-1}"
PATTERN="${PATTERN:-*.txt}"
ABLATION="${ABLATION:-default}"
JAW_MASK_OVERRIDES="${JAW_MASK_OVERRIDES:-configs/loss_overrides/mouth_interior_jaw.json}"
EXTRA_ARGS=(--ablation "$ABLATION" --loss-overrides "$JAW_MASK_OVERRIDES")

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

if [[ "${REVERSE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reverse)
fi

if [[ "${SHUFFLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--shuffle)
  if [[ -n "${SHUFFLE_SEED:-}" ]]; then
    EXTRA_ARGS+=(--shuffle-seed "$SHUFFLE_SEED")
  fi
fi

if [[ $# -gt 0 ]]; then
  EXTRA_ARGS+=("$@")
fi

echo "Repo:       $ROOT"
echo "Rounds:     $ROUNDS"
echo "Reverse:    ${REVERSE:-0} (1 = configs_tmp filename Z→A)"
echo "Shuffle:    ${SHUFFLE:-0} (1 = random config order; SHUFFLE_SEED optional)"
echo "Sleep:      ${SLEEP_SEC}s between rounds"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "Ablation:   $ABLATION"
echo "Jaw mask:   mouth_interior_jaw_only_expression (via $(basename "$JAW_MASK_OVERRIDES"))"
echo "Color expr: stages 2–3 (default); mouth/eye regions gated off (loss_overrides)"
echo "Log root:   /Bean/log/gwangjin/2026/neural_blendshapes_10/<run_name>"
echo ""

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for ((i = 1; i <= ROUNDS; i++)); do
  echo "=============================="
  echo " sweep_configs round ${i}/${ROUNDS}  $(date -Is 2>/dev/null || date)"
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
