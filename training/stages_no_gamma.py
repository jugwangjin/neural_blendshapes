"""
No-gamma schedule: tracker does not train gamma (MP→ICT exponentiation).

Uses ``use_ict_raw_coeffs`` (identity gamma on forward). Pose residual + pose_weight
personalize head pose; ``expr_mlp`` trains from stage 2 alongside GS for a fairer comparison
to the default schedule (gamma in bootstrap/coarse, expr_mlp only in stage 3).
"""

from training.stages import (
    BASIC_LOSS,
    BOOTSTRAP_BASE_LOSS,
    BOOTSTRAP_TEMPLATE_LOSS,
    StageSpec,
    describe_stage_trainables,
    iter_stages,
    stage_at_global_step,
    total_training_steps,
)

_NO_GAMMA = dict(train_gamma=False, use_ict_raw_coeffs=True)

BOOTSTRAP_NO_GAMMA = {**BOOTSTRAP_BASE_LOSS, "w_gamma_prior": 0.0}
BOOTSTRAP_TEMPLATE_NO_GAMMA = {**BOOTSTRAP_TEMPLATE_LOSS, "w_gamma_prior": 0.0}

STAGE_SCHEDULE_NO_GAMMA: list[StageSpec] = [
    StageSpec(
        name="0_precompute",
        steps=0,
        description="Precompute MP/seg/camera caches.",
    ),
    StageSpec(
        name="0_bootstrap_identity",
        steps=2500,
        description=(
            "Bootstrap pose + pose_weight + ICT identity PCA (no gamma); template_mlp frozen; "
            "lmk + mesh_seg."
        ),
        train_pose_residual=True,
        train_pose_weight=True,
        train_template_deformer=False,
        train_ict_identity=True,
        lr_tracker=3e-4,
        lr_pose_weight=1e-3,
        lr_identity=1e-3,
        **_NO_GAMMA,
        **BOOTSTRAP_NO_GAMMA,
    ),
    StageSpec(
        name="1_bootstrap_template",
        steps=5000,
        description=(
            "Bootstrap pose + template_mlp + pose_weight (no gamma); mesh silhouette/seg. "
            "Identity PCA frozen after stage 0."
        ),
        train_pose_residual=True,
        train_pose_weight=True,
        train_template_deformer=True,
        lr_tracker=3e-4,
        lr_pose_weight=1e-3,
        lr_template=2e-4,
        **_NO_GAMMA,
        **BOOTSTRAP_TEMPLATE_NO_GAMMA,
    ),
    StageSpec(
        name="2_coarse_mesh",
        steps=15000,
        description=(
            "Pose + template + pose_weight + expr_mlp + GS (no gamma); RGB/sil/lmk/seg; densify."
        ),
        train_pose_residual=True,
        train_pose_weight=True,
        lr_pose_weight=1e-3,
        train_template_deformer=True,
        train_expression_deform=True,
        lr_expr_deform=2e-5,
        train_gaussian_appearance=True,
        train_color_pose=False,
        train_color_expression=True,
        train_gaussian_geometry=True,
        train_gaussian_h=True,
        sh_degree=3,
        lr_tracker=2e-4,
        lr_template=1e-4,
        **_NO_GAMMA,
        **{
            **BASIC_LOSS,
            "w_gamma_prior": 0.0,
            "w_rgb": 2.0,
            "w_silhouette": 3.0,
            "silhouette_l1": False,
            "silhouette_detach_covariance": False,
            "w_geometry": 0.3,
            "w_opacity_loose": 0.02,
            "w_h": 0.14,
            "h_w_mouth": 0.0,
            "h_w_hair": 0.005,
            "w_mp_lmk": 6.0,
            "w_pie68_jaw": 4.5,
            "w_lip_mouth_leak": 0.0,
            "w_seg": 3.0,
            "seg_l1": False,
            "w_mesh_seg": 1.0,
            "mesh_seg_stop_local": 3000,
            "w_expr_deform_reg": 0.1,
            "w_lpips": 0.01,
            "lpips_start_local": 1,
        },
    ),
    StageSpec(
        name="3_expression_detail",
        steps=15000,
        description=(
            "Tracker frozen; expr_mlp + template + pose_weight + GS detail (no gamma)."
        ),
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
        **_NO_GAMMA,
        **{
            **BASIC_LOSS,
            "w_gamma_prior": 0.0,
            "w_rgb": 2.0,
            "w_silhouette": 3.0,
            "silhouette_l1": False,
            "silhouette_detach_covariance": False,
            "w_geometry": 0.3,
            "w_opacity_loose": 0.02,
            "w_h": 0.10,
            "h_w_mouth": 0.0,
            "h_w_hair": 0.005,
            "w_mp_lmk": 6.0,
            "w_pie68_jaw": 4.0,
            "w_expr_deform_reg": 0.3,
            "w_lpips": 0.01,
            "lpips_start_local": 1,
            "w_lip_mouth_leak": 0.0,
            "w_seg": 3.0,
            "seg_l1": False,
        },
    ),
]
