"""Apply JSON/dict hyperparameter overrides for sweep runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_loss_overrides(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"loss overrides must be a JSON object, got {type(data)}")
    return data


def _apply_basic_loss_to_spec(spec, basic_loss: dict[str, Any]):
    for key, val in basic_loss.items():
        if hasattr(spec, key):
            setattr(spec, key, val)


def resolve_training_schedule(overrides: dict[str, Any] | None):
    """Pick base ``STAGE_SCHEDULE`` from overrides (default, no_gamma, no_gamma_and_pose, additive_gamma)."""
    from training.stages import STAGE_SCHEDULE

    if not overrides:
        return STAGE_SCHEDULE
    key = overrides.get("schedule")
    if key in (None, "", "default"):
        return STAGE_SCHEDULE
    if key == "no_gamma":
        from training.stages_no_gamma import STAGE_SCHEDULE_NO_GAMMA

        return STAGE_SCHEDULE_NO_GAMMA
    if key == "no_gamma_and_pose":
        from training.stages_no_gamma_and_pose import STAGE_SCHEDULE_NO_GAMMA_AND_POSE

        return STAGE_SCHEDULE_NO_GAMMA_AND_POSE
    if key == "additive_gamma":
        from training.stages_additive_gamma import STAGE_SCHEDULE_ADDITIVE_GAMMA

        return STAGE_SCHEDULE_ADDITIVE_GAMMA
    raise ValueError(f"unknown training schedule override: {key!r}")


def apply_loss_overrides(cfg, schedule, overrides: dict[str, Any]):
    """
    ``overrides`` schema::

        {
          "schedule": "no_gamma",
          "config": {"w_scaling": 0.75, "gaussian_max_per_face": 8},
          "basic_loss": {"w_rgb": 4.0, "w_scaling": 0.75},
          "stages": {
            "2_coarse_mesh": {"w_h": 0.0},
            "3_expression_detail": {"w_lpips": 0.005, "lpips_start_local": 10000}
          }
        }
    """
    if not overrides:
        return schedule

    cfg_updates = overrides.get("config", {})
    for key, val in cfg_updates.items():
        setattr(cfg, key, val)

    basic_loss = overrides.get("basic_loss", {})
    apply_basic_to_bootstrap = bool(overrides.get("apply_basic_to_bootstrap", False))
    stage_updates = overrides.get("stages", {})

    new_schedule = []
    for spec in schedule:
        spec = deepcopy(spec)
        if basic_loss and (
            apply_basic_to_bootstrap
            or spec.name not in ("0_bootstrap_identity", "1_bootstrap_template")
        ):
            _apply_basic_loss_to_spec(spec, basic_loss)
        per_stage = stage_updates.get(spec.name, {})
        if per_stage:
            _apply_basic_loss_to_spec(spec, per_stage)
        new_schedule.append(spec)

    return new_schedule
