"""Serialize / restore ``StageSpec`` for checkpoints and render."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from training.stages import StageSpec


def stage_spec_to_dict(spec) -> dict[str, Any]:
    if dataclasses.is_dataclass(spec):
        return {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
    return {"repr": repr(spec)}


def stage_spec_from_dict(data: dict[str, Any]) -> StageSpec:
    fields = {f.name for f in dataclasses.fields(StageSpec)}
    return StageSpec(**{k: v for k, v in data.items() if k in fields})


def _parse_stage_schedule_json(data) -> list[dict]:
    """``STAGE_SCHEDULE.json`` is either a list of specs or ``{stages: [...]}``."""
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        stages = data.get("stages") or data.get("schedule")
        if isinstance(stages, list):
            return [s for s in stages if isinstance(s, dict)]
    return []


def load_stage_spec_from_codes(
    output_root: Path,
    stage_name: str | None = None,
    *,
    final: bool = False,
) -> StageSpec | None:
    codes = Path(output_root) / "codes"
    if not codes.is_dir():
        return None
    for stamp in sorted(codes.iterdir(), reverse=True):
        p = stamp / "STAGE_SCHEDULE.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        stages = _parse_stage_schedule_json(data)
        if not stages:
            continue
        if stage_name is not None:
            for spec_dict in stages:
                if spec_dict.get("name") == stage_name:
                    return stage_spec_from_dict(spec_dict)
        if final:
            last = stages[-1]
            if "name" in last:
                return stage_spec_from_dict(last)
    return None


def load_final_stage_spec_from_codes(output_root: Path) -> StageSpec | None:
    return load_stage_spec_from_codes(output_root, final=True)


def resolve_render_stage_spec(payload, output_root: Path, *, infer_ablation=None):
    """
    Priority: ckpt ``stage_spec`` → codes spec matching ckpt ``stage`` → codes final stage → code fallback.

    Returns ``(spec, source)`` where ``source`` describes which path was taken.
    """
    if payload.get("stage_spec"):
        return stage_spec_from_dict(payload["stage_spec"]), "checkpoint stage_spec"

    ckpt_stage = payload.get("stage")
    if ckpt_stage:
        from_codes = load_stage_spec_from_codes(output_root, str(ckpt_stage))
        if from_codes is not None:
            return from_codes, f"codes/STAGE_SCHEDULE.json stage={ckpt_stage}"

    from_codes = load_final_stage_spec_from_codes(output_root)
    if from_codes is not None:
        return from_codes, "codes/STAGE_SCHEDULE.json (final stage)"

    from training.loss_overrides import resolve_training_schedule

    overrides = load_run_loss_overrides(output_root, infer_ablation=infer_ablation)
    schedule = resolve_training_schedule(overrides or None)
    print(
        "render spec: WARNING — no checkpoint stage_spec or codes/STAGE_SCHEDULE.json; "
        "using current-code schedule"
    )
    return schedule[-1], "current-code schedule (fallback)"


def load_run_loss_overrides(output_root: Path, *, infer_ablation=None) -> dict:
    from training.loss_overrides import load_loss_overrides

    codes = Path(output_root) / "codes"
    if codes.is_dir():
        for stamp in sorted(codes.iterdir(), reverse=True):
            p = stamp / "loss_overrides.json"
            if p.is_file():
                return load_loss_overrides(p)
    if infer_ablation is not None:
        ablation = infer_ablation(output_root)
        if ablation != "default":
            return {"schedule": ablation}
    return {}


def final_stage_spec(output_root: Path, *, overrides: dict | None = None, infer_ablation=None):
    if overrides is None:
        overrides = load_run_loss_overrides(output_root, infer_ablation=infer_ablation)
    from training.loss_overrides import resolve_training_schedule

    schedule = resolve_training_schedule(overrides or None)
    return schedule[-1]
