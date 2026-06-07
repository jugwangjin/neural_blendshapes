"""Training run completion checks (shared by run_config.py / sweep_configs.py)."""

from __future__ import annotations

from pathlib import Path

LOG_DIR_2026 = Path("/Bean/log/gwangjin/2026")
# Default sweep (unchanged flat layout: ``.../neural_blendshapes_10/<run_name>``).
LOG_ROOT_BASE = LOG_DIR_2026 / "neural_blendshapes_10"
ABLATION_DEFAULT = "default"
FINAL_STAGE_NAME = "3_expression_detail"

ABLATION_LOG_ROOTS: dict[str, Path] = {
    "default": LOG_ROOT_BASE,
    "no_gamma": LOG_DIR_2026 / "neural_blendshapes_10_no_gamma",
    "no_gamma_and_pose": LOG_DIR_2026 / "neural_blendshapes_10_no_gamma_and_pose",
    "additive_gamma": LOG_DIR_2026 / "neural_blendshapes_10_additive_gamma",
    "opacity_decay": LOG_DIR_2026 / "neural_blendshapes_10_opacity_decay",
}


def log_root_for_ablation(ablation: str | None = None) -> Path:
    key = (ablation or ABLATION_DEFAULT).strip() or ABLATION_DEFAULT
    if key not in ABLATION_LOG_ROOTS:
        raise ValueError(
            f"unknown ablation {key!r}; expected one of {sorted(ABLATION_LOG_ROOTS)}"
        )
    return ABLATION_LOG_ROOTS[key]


def output_root_for_run(
    run_name: str,
    *,
    log_root: Path | str | None = None,
    ablation: str | None = None,
) -> Path:
    root = Path(log_root) if log_root is not None else log_root_for_ablation(ablation)
    return root / run_name


def is_run_complete(output_root: Path, *, final_stage: str = FINAL_STAGE_NAME) -> bool:
    """
    True when the final stage-end checkpoint exists under ``output_root/checkpoints``.
    """
    ckpt_dir = Path(output_root) / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    pattern = f"stage_{final_stage}_end_step_*.pt"
    return any(ckpt_dir.glob(pattern))
