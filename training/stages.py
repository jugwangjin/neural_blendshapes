"""
Training schedule (precompute → bootstrap pose → coarse mesh → expression detail).

Loss stack on stages 2–3: RGB, silhouette, mp_lmk, pie68_jaw, gamma, GT image-space h,
geometry, opacity, pose/template regs. Stage 1 bootstrap: landmarks only (no RGB/silhouette —
silhouette needs template deformation). No ``w_seg`` / semantic render / expression-gap losses.

  Stage 1 — Landmark pose only (R+t+global scale; mp_lmk + pie68_jaw).
            No gsplat rasterization or surface-Gaussian forward (fast path).

  Stage 2 — Tracker + template + pose weight + surface GS (small geometry LR).

  Stage 3 — Expression residual + surface GS detail (tracker frozen; template small LR).

  View dependency is disabled. Color is conditioned on pose/expression displacement.
"""

from dataclasses import dataclass
from typing import Optional


# Shared supervision for all train stages (no separate smoke stage).
BASIC_LOSS = dict(
    w_rgb=1.0,
    w_silhouette=10.0,
    w_mp_lmk=50.0,
    w_pie68_jaw=25.0,
    w_h=0.25,
    w_geometry=0.01,
    w_scaling=0.01,
    w_opacity=0.0,
    w_opacity_decay=0.0,
    w_gamma_prior=0.5,
    w_pose_prior=0.01,
    w_template_smooth=0.005,
    w_seg=0.0,
    w_sem_anchor=0.0,
    w_expr_neutral=0.0,
    w_expr_leak=0.0,
    w_expr_amp=0.0,
)

# Bootstrap: pose from 2D/3D landmarks only (no template / GS appearance yet).
BOOTSTRAP_LOSS = {
    **BASIC_LOSS,
    "w_rgb": 0.0,
    "w_silhouette": 0.0,
    "w_h": 0.0,
    "w_geometry": 0.0,
    "w_scaling": 0.0,
    "w_opacity": 0.0,
    "w_opacity_decay": 0.0,
}


@dataclass
class StageSpec:
    name: str
    steps: int
    description: str = ""

    train_tracker: bool = False
    train_gamma: bool = False
    fix_gamma_at_one: bool = False
    train_pose_residual: bool = False
    train_pose_scale: bool = True
    apply_pose_scale: bool = True
    pose_rotate_about_centroid: bool = False
    pose_weight_one: bool = False
    pose_zero_tz: bool = False
    train_pose_weight: bool = False
    train_template_deformer: bool = False
    train_expression_deform: bool = False

    train_gaussian_appearance: bool = False
    train_color_pose: bool = False
    train_color_expression: bool = False
    train_gaussian_geometry: bool = False
    train_gaussian_semantic: bool = False
    geometry_lr_scale: float = 1.0
    sh_degree: Optional[int] = None

    w_rgb: float = 0.0
    w_mp_lmk: float = 0.0
    w_pie68_jaw: float = 0.0
    w_silhouette: float = 10.0
    w_mask: float = 0.0  # alias → w_silhouette in train_losses if w_silhouette unset
    w_seg: float = 0.0
    w_h: float = 0.0
    w_geometry: float = 0.0
    w_scaling: float = 0.0
    w_opacity: float = 0.0
    w_gamma_prior: float = 0.0
    h_skin_sigma: float = 0.002
    h_sigma_brow: float = 0.004
    h_sigma_misc: float = 0.010
    h_sigma_mouth: float = 0.008
    h_w_skin: float = 1.0
    h_w_eye: float = 1.0
    h_w_brow: float = 0.45
    h_w_misc: float = 0.12
    h_w_mouth: float = 0.28
    h_alpha_min: float = 0.08
    geometry_max_scale: float = 0.008
    thresh_scaling_max: float = 0.008
    thresh_scaling_ratio: float = 10.0
    opacity_target: float = 1.0
    opacity_w_skin: float = 1.0
    opacity_w_other: float = 0.05
    w_pose_prior: float = 0.0
    w_pose_tz: float = 0.0
    w_expr_deform_reg: float = 0.0
    w_expr_neutral: float = 0.0
    w_expr_leak: float = 0.0
    w_expr_amp: float = 0.0
    w_sem_anchor: float = 0.0
    w_template_smooth: float = 0.0
    w_opacity_decay: float = 0.0

    lr_tracker: float = 1e-4
    lr_pose_weight: float = 5e-5
    lr_template: float = 5e-5
    lr_expr_deform: float = 5e-5
    lr_gaussian_h: float = 5e-4
    lr_gaussian_color: float = 1e-2
    lr_gaussian_opacity: float = 5e-3
    lr_gaussian_scale: float = 5e-3


STAGE_SCHEDULE: list[StageSpec] = [
    StageSpec(
        name="0_precompute",
        steps=0,
        description="Precompute MP/seg/camera caches.",
    ),
    StageSpec(
        name="1_bootstrap_pose",
        steps=5000,
        description=(
            "Landmark pose only: R+t+global scale (world origin), tz free; "
            "no expression deform / template / GS training."
        ),
        fix_gamma_at_one=True,
        train_pose_residual=True,
        pose_weight_one=True,
        pose_zero_tz=False,
        lr_tracker=2e-4,
        **BOOTSTRAP_LOSS,
    ),
    StageSpec(
        name="2_coarse_mesh",
        steps=15000,
        description=(
            "Tracker + template + pose weight; surface RGB/silhouette/lmk; "
            "optional global scale; GT image-space h (no sem render)."
        ),
        train_tracker=True,
        train_gamma=True,
        train_pose_residual=True,
        train_pose_scale=True,
        apply_pose_scale=True,
        train_pose_weight=True,
        pose_zero_tz=False,
        train_template_deformer=True,
        train_gaussian_appearance=True,
        train_color_pose=True,
        train_color_expression=False,
        train_gaussian_geometry=True,
        geometry_lr_scale=0.1,
        sh_degree=None,
        lr_tracker=1e-4,
        lr_template=1.5e-4,
        lr_gaussian_h=2.5e-4,
        **{**BASIC_LOSS, "w_opacity": 1.0},
    ),
    StageSpec(
        name="3_expression_detail",
        steps=15000,
        description=(
            "Tracker frozen; template + expression residual + surface GS "
            "(template LR << stage 1 so coeffs stay stable while mesh refines)."
        ),
        train_template_deformer=True,
        train_expression_deform=True,
        train_gaussian_appearance=True,
        train_color_pose=True,
        train_color_expression=True,
        train_gaussian_geometry=True,
        geometry_lr_scale=0.15,
        sh_degree=None,
        lr_template=3e-5,
        lr_expr_deform=1.5e-4,
        lr_gaussian_h=2.0e-4,
        lr_gaussian_color=8e-3,
        lr_gaussian_opacity=4e-3,
        **{**BASIC_LOSS, "w_opacity": 0.1},
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
