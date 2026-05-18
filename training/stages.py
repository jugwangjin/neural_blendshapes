"""
2-stage training schedule (+ optional Stage 2A/2B warmup).

Stage 1: tracker + pose + template deformer (coarse geometry)
Stage 2A: expression deformer + Gaussian appearance (geometry low LR)
Stage 2B: expression (low LR) + full Gaussian detail
"""

from dataclasses import dataclass


@dataclass
class StageSpec:
    name: str
    steps: int
    description: str = ""

    # Mesh / tracking
    train_tracker: bool = False
    train_gamma: bool = False
    fix_gamma_at_one: bool = False
    train_pose_residual: bool = False
    train_pose_weight: bool = False
    train_template_deformer: bool = False
    train_expression_deform: bool = False
    train_eye_gaze: bool = False

    # Gaussian subsets
    train_gaussian_appearance: bool = False
    train_gaussian_geometry: bool = False
    train_gaussian_semantic: bool = False
    geometry_lr_scale: float = 1.0

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
    lr_gaussian_uv: float = 1e-3
    lr_gaussian_h: float = 5e-4
    lr_gaussian_color: float = 1e-2
    lr_gaussian_opacity: float = 5e-3
    lr_gaussian_scale: float = 1e-3
    lr_eye_gaze: float = 1e-3

    mesh_update_interval: int = 1


STAGE_SCHEDULE: list[StageSpec] = [
    StageSpec(
        name="0_precompute",
        steps=0,
        description="Precompute MP/seg/camera caches.",
    ),
    StageSpec(
        name="1_coarse_geometry",
        steps=15000,
        description="Tracker + pose + template deformer; Gaussian position frozen.",
        train_tracker=True,
        train_gamma=True,
        train_pose_residual=True,
        train_pose_weight=True,
        train_template_deformer=True,
        train_gaussian_appearance=True,
        train_eye_gaze=False,
        w_mp_lmk=100.0,
        w_iris=50.0,
        w_mask=10.0,
        w_seg=2.0,
        w_rgb=0.3,
        w_h=0.5,
        w_gamma_prior=5.0,
        w_pose_prior=1.0,
        w_gaze_residual=0.1,
        w_template_smooth=0.1,
    ),
    StageSpec(
        name="2A_expression_warmup",
        steps=5000,
        description="Expression deformer on; Gaussian color/opacity; uv/h low LR.",
        train_expression_deform=True,
        train_gaussian_appearance=True,
        train_gaussian_geometry=True,
        train_gaussian_semantic=True,
        geometry_lr_scale=0.1,
        train_eye_gaze=True,
        w_rgb=0.8,
        w_mask=5.0,
        w_seg=2.0,
        w_sem_anchor=0.1,
        w_mp_lmk=40.0,
        w_iris=25.0,
        w_h=0.4,
        w_expr_neutral=0.5,
        w_expr_leak=0.2,
        w_expr_amp=0.1,
        w_scale=0.005,
        w_opacity=0.005,
        lr_expr_deform=5e-5,
        lr_gaussian_h=5e-5,
        lr_gaussian_uv=1e-4,
    ),
    StageSpec(
        name="2B_gaussian_detail",
        steps=35000,
        description="Full Gaussian detail; template/tracker frozen; expr low LR.",
        train_expression_deform=True,
        train_gaussian_appearance=True,
        train_gaussian_geometry=True,
        train_gaussian_semantic=True,
        train_eye_gaze=True,
        w_rgb=1.0,
        w_mask=5.0,
        w_seg=2.0,
        w_sem_anchor=0.05,
        w_mp_lmk=15.0,
        w_iris=15.0,
        w_h=0.5,
        w_expr_neutral=0.3,
        w_expr_leak=0.15,
        w_expr_amp=0.08,
        w_scale=0.01,
        w_opacity=0.01,
        w_eye_uv_barrier=0.001,
        lr_expr_deform=1e-5,
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
