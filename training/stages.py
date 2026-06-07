"""
Training schedule (precompute → bootstrap identity → bootstrap template → coarse mesh → expression detail).

  Stage 0 — Gamma + pose + pose_weight + identity PCA; template frozen; lmk/mesh_seg.

  Stage 1 — Gamma + pose + template_mlp + pose_weight; mesh sil/seg; identity frozen.

  Stage 2 — Full tracker + template + GS (RGB/sil/lmk/seg); densify; pose/scale still trainable.

  Stage 3 — Tracker pose/gamma frozen; expr_mlp + template + pose_weight + GS detail.
"""

from dataclasses import dataclass
from typing import Optional


# Shared supervision for all train stages (no separate smoke stage).
# Photometric / mask weights aligned with GaussianBlendshapes config_blendshapes.py:
#   (1-λ)L1+λ(1-SSIM) at λ=0.2; alpha L2 (GB ``(α-mask)²``; GB weight alpha_loss=10).
BASIC_LOSS = dict(
    w_rgb=2.0,
    rgb_ssim_lambda=0.2,
    w_silhouette=3.0,
    w_mesh_silhouette=0.0,
    w_mp_lmk=8.0,  # MP 478 on mesh UV (smooth L1 / Charbonnier)
    w_pie68_jaw=4.0,  # PIE jaw 0:landmark_start only (lips: w_mp_lmk)
    w_h=0.15,
    w_geometry=0.0,  # ReLU(max σ - max_scale)²; only oversized splats (GB has no separate w_scaling)
    w_opacity=0.0,
    w_opacity_loose=0.0,
    w_opacity_headneck=0.00,
    w_opacity_decay=0.0,
    w_face_region=0.0,
    lambda_sparsity=0.0,
    w_lpips=0.0,
    w_normal=0.0,
    w_lip_mouth_leak=0.0,
    w_gamma_prior=0.25,
    w_pose_prior=0.01,
    w_template_smooth=3e-1,
    w_template_laplacian=0.0,
    w_template_scale_prior=0.0,
    w_seg=0.0,
    w_expr_neutral=0.0,
    w_expr_leak=0.0,
    w_expr_amp=0.0,
    w_color_expr_sparse=0.0,
    w_color_expr_group_sparse=0.0,
    w_color_expr_per_gaussian=0.0,
)

# Bootstrap shared (landmarks + pose prior; no GS). Mesh silhouette only in template stage.
BOOTSTRAP_BASE_LOSS = {
    **BASIC_LOSS,
    "w_mp_lmk": 14.0,
    "w_pie68_jaw": 5.0,
    "w_rgb": 0.0,
    "w_silhouette": 0.0,
    "w_mesh_silhouette": 0.0,
    "w_mesh_seg": 5,
    "mesh_backface_curl_weight": 0.0,
    "w_gamma_prior": 0.25,
    "w_normal": 0.0,
    "w_h": 0.0,
    "w_geometry": 0.0,
    "w_opacity": 0.0,
    "w_opacity_loose": 0.0,
    "w_opacity_headneck": 0.0,
    "w_opacity_decay": 0.0,
    "w_face_region": 0.0,
    "lambda_sparsity": 0.0,
}

BOOTSTRAP_TEMPLATE_LOSS = {
    **BOOTSTRAP_BASE_LOSS,
    "w_mesh_silhouette": 0.25,
    "w_mesh_seg": 8,
    "mesh_backface_curl_weight": 0.25,
    "w_normal": 0.0,
    "w_gamma_prior": 0.25,
}


@dataclass
class StageSpec:
    name: str
    steps: int
    description: str = ""

    train_tracker: bool = False
    train_gamma: bool = False
    train_pose_residual: bool = False
    # Optional: force MP→ICT raw coeffs in forward (skip gamma); default False — gamma trains from stage 0.
    use_ict_raw_coeffs: bool = False
    # ``coeffs = ict_raw + head_gamma`` instead of ``ict_raw ** gamma``.
    additive_gamma_correction: bool = False
    # Inference / personalization viz only — never set True on training stages (see tracker forward).
    fix_gamma_at_one: bool = False
    train_pose_scale: bool = True
    # Learn ``tracker.global_translation`` (3D) instead of ``head_pose`` global slice (no_gamma_and_pose).
    train_global_translation: bool = False
    use_global_translation_param: bool = False
    apply_pose_scale: bool = True
    pose_rotate_about_centroid: bool = False
    pose_weight_one: bool = False
    pose_zero_tz: bool = False
    train_pose_weight: bool = False
    train_template_deformer: bool = False
    train_ict_identity: bool = False
    train_expression_deform: bool = False

    train_gaussian_appearance: bool = False
    train_color_pose: bool = False
    train_color_expression: bool = False
    train_gaussian_geometry: bool = False
    train_gaussian_h: bool = False
    geometry_lr_scale: float = 1.0
    sh_degree: Optional[int] = None

    w_rgb: float = 0.0
    rgb_ssim_lambda: float = 0.2
    w_mp_lmk: float = 0.0
    w_pie68_jaw: float = 0.0
    w_silhouette: float = 10.0
    silhouette_l1: bool = False
    silhouette_detach_covariance: bool = False
    w_mesh_silhouette: float = 0.0
    w_mesh_seg: float = 0.0
    mesh_seg_stop_local: int = 0  # 0 = full stage; else mesh seg only for stage_local <= this
    mesh_backface_curl_weight: float = 0.0
    w_mask: float = 0.0  # alias → w_silhouette in train_losses if w_silhouette unset
    w_seg: float = 0.0
    seg_l1: bool = False
    w_h: float = 0.0
    w_geometry: float = 0.0
    w_opacity: float = 0.0
    w_opacity_loose: float = 0.0
    w_opacity_headneck: float = 0.0
    w_face_region: float = 0.0
    face_region_alpha_min: float = 0.02
    lambda_sparsity: float = 0.0
    w_lpips: float = 0.0
    w_normal: float = 0.0
    w_lip_mouth_leak: float = 0.0
    lpips_start_local: int = 0
    lpips_net: str = "alex"
    w_gamma_prior: float = 0.0
    # Per-tier h_reg: h_w * accum_h² only (no sigma). High → mesh-stick.
    h_w_skin: float = 2.4
    h_w_nose: float = 1.4
    h_w_eye: float = 2.8
    h_w_brow: float = 1.4
    h_w_neck: float = 1.4
    h_w_cloth: float = 0.1
    h_w_misc: float = 1.4
    h_w_mouth: float = 0.0
    h_w_hair: float = 0.005
    h_w_glasses: float = 0.006
    h_teeth_h_loss_scale: float = 1.0
    h_eye_occlusion_h_loss_scale: float = 2.5
    h_alpha_min: float = 0.08
    geometry_max_scale: float = 0.004
    thresh_scaling_max: float = 0.008
    thresh_scaling_ratio: float = 10.0
    opacity_target: float = 1.0
    opacity_loose_target: float = 1.0
    opacity_w_skin: float = 1.0
    opacity_w_other: float = 0.05
    w_pose_prior: float = 0.0
    w_pose_tz: float = 0.0
    w_expr_deform_reg: float = 0.0
    w_expr_neutral: float = 0.0
    w_expr_leak: float = 0.0
    w_expr_amp: float = 0.0
    w_color_expr_sparse: float = 0.0
    w_color_expr_group_sparse: float = 0.0
    w_color_expr_per_gaussian: float = 0.0
    w_template_smooth: float = 0.0
    w_template_laplacian: float = 0.0
    w_template_scale_prior: float = 0.0
    w_identity_prior: float = 0.0
    w_opacity_decay: float = 0.0

    lr_tracker: float = 1e-4
    lr_pose_weight: float = 5e-5
    lr_template: float = 5e-5
    lr_identity: float = 1e-2
    lr_expr_deform: float = 5e-5
    # GaussianBlendshapes config_blendshapes.py: position/feature/opacity/scaling/rotation_lr
    lr_gaussian_h: float = 1.6e-4  # position_lr_init; bary_uv uses same
    lr_gaussian_color: float = 2.5e-3  # feature_lr (DC/RGB); color_pose & color_expression use /4 in apply.py
    lr_gaussian_opacity: float = 5e-2  # opacity_lr
    lr_gaussian_scale: float = 5e-3  # scaling_lr
    lr_gaussian_rotation: float = 1e-3  # rotation_lr
    # GB position_lr_final / position_lr_init (= 0.01); stage-local exp decay on geometry params.
    geometry_lr_decay: bool = False
    geometry_lr_decay_final_mult: float = 0.01
    geometry_lr_decay_start_frac: float = 0.0  # 0 = from step 0; 0.5 = decay only in 2nd half


STAGE_SCHEDULE: list[StageSpec] = [
    # Global: 2500 + 5000 bootstrap → 7500; 2_coarse_mesh 7501–22500; 3_expression 22501–42500.
    # Densify (config): 2 grow [1,15000]; 3 grow [1,5000], cleanup prune [5000,10000].
    StageSpec(
        name="0_precompute",
        steps=0,
        description="Precompute MP/seg/camera caches.",
    ),
    StageSpec(
        name="0_bootstrap_identity",
        steps=2500,
        description=(
            "Bootstrap gamma + pose + pose_weight + ICT identity PCA; template_mlp frozen; "
            "lmk + mesh_seg (no mesh silhouette)."
        ),
        train_gamma=True,
        train_pose_residual=True,
        train_pose_weight=True,
        train_template_deformer=False,
        train_ict_identity=True,
        lr_tracker=3e-4,
        lr_pose_weight=1e-3,
        lr_identity=1e-3,
        **BOOTSTRAP_BASE_LOSS,
    ),
    StageSpec(
        name="1_bootstrap_template",
        steps=5000,
        description=(
            "Bootstrap gamma + pose + template_mlp + pose_weight; mesh silhouette/seg + normal + backface curl. "
            "Identity PCA frozen after stage 0."
        ),
        train_gamma=True,
        train_pose_residual=True,
        train_pose_weight=True,
        train_template_deformer=True,
        lr_tracker=3e-4,
        lr_pose_weight=1e-3,
        lr_template=2e-4,  # first template_mlp fit; > stage-0 tracker (template was frozen)
        **BOOTSTRAP_TEMPLATE_LOSS,
    ),
    StageSpec(
        name="2_coarse_mesh",
        steps=15000,
        description=(
            "Tracker (gamma+pose) + template + pose_weight + GS; RGB/sil/lmk/seg; densify; "
            "mesh_seg off after local 6000."
        ),
        train_gamma=True,
        train_pose_residual=True,
        train_pose_weight=True,
        lr_pose_weight=1e-3,
        train_template_deformer=True,
        train_gaussian_appearance=True,
        train_color_pose=False,
        train_color_expression=True,
        train_gaussian_geometry=True,
        train_gaussian_h=True,
        sh_degree=3,
        lr_tracker=2e-4,
        lr_template=1e-4,
        **{
            **BASIC_LOSS,
            "w_rgb": 2.0,
            "w_silhouette": 5.0,
            "silhouette_l1": False,
            "silhouette_detach_covariance": False,
            "w_geometry": 0.01,
            "w_opacity_loose": 0.0,
            "w_h": 0.04,
            "h_w_mouth": 0.0,
            "h_w_hair": 0.0005,
            "w_mp_lmk": 6.0,
            "w_pie68_jaw": 4.5,
            "w_lip_mouth_leak": 0.25,
            "w_seg": 5.0,
            "seg_l1": False,
            "w_mesh_seg": 1.0,
            "mesh_seg_stop_local": 1500,
            "w_lpips": 0.01,
            "lpips_start_local": 1,
        },
    ),
    StageSpec(
        name="3_expression_detail",
        steps=15000,
        description=(
            "Tracker pose/gamma/scale frozen; expr_mlp + template + pose_weight + GS; "
            "expression residual + appearance/geometry detail."
        ),
        train_gamma=False,
        train_pose_residual=False,
        train_pose_scale=False,
        train_pose_weight=True,
        lr_pose_weight=1e-3,
        train_template_deformer=True,
        train_expression_deform=True,
        train_gaussian_appearance=True,
        train_color_pose=False,
        train_color_expression=True,
        train_gaussian_geometry=True,
        train_gaussian_h=True,
        sh_degree=3,
        lr_template=3e-5,
        lr_expr_deform=2e-5,
        geometry_lr_decay=True,
        geometry_lr_decay_start_frac=0.0,
        **{
            **BASIC_LOSS,
            "w_rgb": 2.0,
            "w_silhouette": 5.0,
            "silhouette_l1": False,
            "silhouette_detach_covariance": False,
            "w_geometry": 0.01,
            "w_opacity_loose": 0.0,
            "w_h": 0.04,
            "h_w_mouth": 0.0,
            "h_w_hair": 0.0005,
            "w_mp_lmk": 6.0,
            "w_pie68_jaw": 4.0,
            "w_expr_deform_reg": 0.3,
            "w_lpips": 0.01,
            "lpips_start_local": 1,
            "w_lip_mouth_leak": 0.25,
            "w_seg": 5.0,
            "seg_l1": False,
        },
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


def describe_stage_trainables(spec: StageSpec) -> list[str]:
    """Human-readable trainable modules for logging (mirrors ``apply_stage_requires_grad``)."""
    if spec.steps <= 0:
        return []
    out: list[str] = []
    if spec.train_tracker:
        out.append("tracker: expr_trunk + head_gamma + pose_trunk + head_pose")
    else:
        parts = []
        if spec.train_gamma:
            parts.append("expr_trunk + head_gamma")
        if spec.train_pose_residual:
            parts.append("pose_trunk + head_pose")
        if parts:
            out.append("tracker: " + ", ".join(parts))
        else:
            out.append("tracker: frozen (inference only)")
    if getattr(spec, "use_ict_raw_coeffs", False):
        out.append("tracker forward: coeffs=ict_raw (bootstrap)")
    if getattr(spec, "additive_gamma_correction", False):
        out.append("tracker forward: additive gamma residual (not pow)")
    if spec.train_pose_scale:
        out.append("tracker.log_pose_scale (global scale)")
    if getattr(spec, "train_global_translation", False):
        out.append("tracker.global_translation (3D, subject-global)")
    if getattr(spec, "use_global_translation_param", False):
        out.append("forward: global_translation param (not head_pose[9:12])")
    if spec.train_pose_weight:
        out.append("deformer.pose_weight_net")
    if spec.train_template_deformer:
        out.append("deformer.template_mlp")
    if getattr(spec, "train_ict_identity", False):
        out.append("deformer.identity_weights")
    if spec.train_expression_deform:
        out.append("deformer.expr_mlp")
    gs = []
    if spec.train_gaussian_appearance:
        gs.append("color/opacity")
    if getattr(spec, "train_color_pose", False):
        gs.append("color_pose")
    if getattr(spec, "train_color_expression", False):
        gs.append("color_expression")
    if spec.train_gaussian_geometry:
        gs.append("bary/scale/rot")
    if spec.train_gaussian_h:
        gs.append("h")
    if gs:
        out.append("surface GS: " + ", ".join(gs))
    return out
