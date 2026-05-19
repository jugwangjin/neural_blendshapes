"""
Training schedule (mesh → expression → view-independent GS → view-dependent appearance).

  Stage 1 — Tracker + template MLP + pose weight; Gaussian color/opacity + small ``h``;
             eye gaze UV slide; ``w_h`` + template reg steer mesh identity.

  Stage 2A — Tracker/template frozen; learn support-gated expression residual + ``h`` detail;
             eye gaze refine/slide still on.

  Stage 2B — All mesh frozen; view-independent Gaussian geometry + appearance (+ accessory).

  Stage 3 — Everything frozen; view-dependent appearance only (raise ``sh_degree`` when supported).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StageSpec:
    name: str
    steps: int
    description: str = ""

    train_tracker: bool = False
    train_gamma: bool = False
    fix_gamma_at_one: bool = False
    train_pose_residual: bool = False
    train_pose_weight: bool = False
    train_template_deformer: bool = False
    train_expression_deform: bool = False
    train_eye_gaze: bool = False

    train_gaussian_appearance: bool = False
    train_gaussian_geometry: bool = False
    train_gaussian_semantic: bool = False
    train_accessory: bool = False
    geometry_lr_scale: float = 1.0
    sh_degree: Optional[int] = None

    w_rgb: float = 0.0
    w_mp_lmk: float = 0.0
    w_iris: float = 0.0
    w_mask: float = 0.0
    w_seg: float = 0.0
    w_h: float = 0.0
    w_eye_uv_barrier: float = 0.0
    w_scale: float = 0.0
    w_opacity: float = 0.0
    w_gamma_prior: float = 0.0
    w_pose_prior: float = 0.0
    w_gaze_residual: float = 0.0
    w_expr_deform_reg: float = 0.0
    w_expr_neutral: float = 0.0
    w_expr_leak: float = 0.0
    w_expr_amp: float = 0.0
    w_sem_anchor: float = 0.0
    w_template_smooth: float = 0.0

    lr_tracker: float = 1e-4
    lr_pose_weight: float = 5e-5
    lr_template: float = 5e-5
    lr_expr_deform: float = 5e-5
    lr_gaussian_h: float = 5e-4
    lr_gaussian_color: float = 1e-2
    lr_gaussian_opacity: float = 5e-3
    lr_gaussian_scale: float = 1e-3
    lr_eye_uv: float = 1e-3
    lr_accessory: float = 1e-3
    lr_eye_gaze: float = 1e-3


STAGE_SCHEDULE: list[StageSpec] = [
    StageSpec(
        name="0_precompute",
        steps=0,
        description="Precompute MP/seg/camera caches.",
    ),
    StageSpec(
        name="1_coarse_mesh",
        steps=15000,
        description=(
            "Tracker + template + pose weight; eye gaze slide; RGB/opacity; "
            "small h (distance prior pulls mesh via deformer)."
        ),
        train_tracker=True,
        train_gamma=True,
        train_pose_residual=True,
        train_pose_weight=True,
        train_template_deformer=True,
        train_eye_gaze=True,
        train_gaussian_appearance=True,
        train_gaussian_geometry=True,
        geometry_lr_scale=0.08,
        sh_degree=None,
        w_mp_lmk=100.0,
        w_iris=50.0,
        w_mask=10.0,
        w_seg=2.0,
        w_rgb=0.35,
        w_h=0.6,
        w_gamma_prior=5.0,
        w_pose_prior=1.0,
        w_gaze_residual=0.1,
        w_template_smooth=0.15,
        lr_gaussian_h=8e-5,
        lr_eye_uv=8e-5,
        lr_eye_gaze=1e-3,
    ),
    StageSpec(
        name="2A_expression",
        steps=8000,
        description=(
            "Tracker/template frozen; expression residual + h detail; "
            "blendshape gap; eye slide on."
        ),
        train_expression_deform=True,
        train_eye_gaze=True,
        train_gaussian_appearance=True,
        train_gaussian_geometry=True,
        train_gaussian_semantic=True,
        geometry_lr_scale=0.15,
        sh_degree=None,
        w_rgb=0.85,
        w_mask=5.0,
        w_seg=2.0,
        w_sem_anchor=0.1,
        w_mp_lmk=35.0,
        w_iris=25.0,
        w_h=0.45,
        w_expr_neutral=0.5,
        w_expr_leak=0.2,
        w_expr_amp=0.1,
        w_gaze_residual=0.05,
        w_scale=0.006,
        w_opacity=0.006,
        lr_expr_deform=5e-5,
        lr_gaussian_h=6e-5,
        lr_eye_gaze=5e-4,
    ),
    StageSpec(
        name="2B_geometry_detail",
        steps=30000,
        description=(
            "Mesh stack frozen; view-independent Gaussian h/scale/color/opacity; "
            "accessory optional."
        ),
        train_gaussian_appearance=True,
        train_gaussian_geometry=True,
        train_accessory=True,
        geometry_lr_scale=1.0,
        sh_degree=None,
        w_rgb=1.0,
        w_mask=5.0,
        w_seg=1.5,
        w_mp_lmk=12.0,
        w_iris=12.0,
        w_h=0.55,
        w_scale=0.01,
        w_opacity=0.01,
        w_eye_uv_barrier=0.001,
        lr_gaussian_h=4e-4,
        lr_gaussian_color=8e-3,
        lr_gaussian_opacity=4e-3,
    ),
    StageSpec(
        name="3_view_appearance",
        steps=8000,
        description=(
            "All geometry/mesh frozen; view-dependent appearance "
            "(set sh_degree>0 when color channels support SH)."
        ),
        train_gaussian_appearance=True,
        sh_degree=None,
        w_rgb=1.0,
        w_seg=1.0,
        w_mp_lmk=4.0,
        w_iris=4.0,
        w_mask=3.0,
        lr_gaussian_color=4e-3,
        lr_gaussian_opacity=2e-3,
    ),
]


def total_training_steps(schedule=None):
    schedule = schedule or STAGE_SCHEDULE
    return sum(s.steps for s in schedule if s.steps > 0)


def stage_at_global_step(global_step, schedule=None):
    schedule = schedule or STAGE_SCHEDULE
    t = 0
    for i, spec in enumerate(schedule):
        if spec.steps <= 0:
            continue
        if global_step < t + spec.steps:
            return i, spec, global_step - t
        t += spec.steps
    i = len(schedule) - 1
    spec = schedule[-1]
    return i, spec, max(0, global_step - t)


def iter_stages(schedule=None):
    schedule = schedule or STAGE_SCHEDULE
    offset = 0
    for i, spec in enumerate(schedule):
        if spec.steps <= 0:
            continue
        yield i, spec, offset, offset + spec.steps
        offset += spec.steps
