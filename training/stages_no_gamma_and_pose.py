"""
No-gamma, no-pose schedule: tracker does not train gamma or per-frame pose MLP.

Forward uses ``use_ict_raw_coeffs``. Rotation / local translation follow frozen MP + zero-init
``head_pose`` (no trunk grad). Subject-global rigid fit uses learnable parameters only:

  - ``log_pose_scale`` (1D global scale)
  - ``global_translation`` (3D; ``use_global_translation_param`` — not ``head_pose[9:12]``)

Other schedules keep per-frame ``translation_global`` from ``head_pose`` when pose trains.
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

_NO_TRACKER_PERSONAL = dict(
    train_gamma=False,
    use_ict_raw_coeffs=True,
    train_pose_residual=False,
    train_pose_weight=False,
)

# Subject-global rigid (scale + translation param); not pose trunk / head_pose.
_GLOBAL_RIGID_TRAIN = dict(
    train_pose_scale=True,
    train_global_translation=True,
    use_global_translation_param=True,
    apply_pose_scale=True,
)

_BOOTSTRAP_RIGID = {**_NO_TRACKER_PERSONAL, **_GLOBAL_RIGID_TRAIN, "lr_tracker": 3e-4}
_COARSE_RIGID = {**_NO_TRACKER_PERSONAL, **_GLOBAL_RIGID_TRAIN, "lr_tracker": 2e-4}
_STAGE3_FROZEN_RIGID = {
    **_NO_TRACKER_PERSONAL,
    "train_pose_scale": False,
    "train_global_translation": False,
    "use_global_translation_param": True,
    "apply_pose_scale": True,
}

BOOTSTRAP_NO_GP = {**BOOTSTRAP_BASE_LOSS, "w_gamma_prior": 0.0}
BOOTSTRAP_TEMPLATE_NO_GP = {**BOOTSTRAP_TEMPLATE_LOSS, "w_gamma_prior": 0.0}

STAGE_SCHEDULE_NO_GAMMA_AND_POSE: list[StageSpec] = [
    StageSpec(
        name="0_precompute",
        steps=0,
        description="Precompute MP/seg/camera caches.",
    ),
    StageSpec(
        name="0_bootstrap_identity",
        steps=2500,
        description=(
            "Bootstrap identity PCA + global scale/translation params (no gamma, no pose MLP); "
            "template_mlp frozen; lmk + mesh_seg."
        ),
        train_template_deformer=False,
        train_ict_identity=True,
        lr_identity=1e-3,
        **_BOOTSTRAP_RIGID,
        **BOOTSTRAP_NO_GP,
    ),
    StageSpec(
        name="1_bootstrap_template",
        steps=5000,
        description=(
            "Bootstrap template_mlp + global scale/translation (no gamma, no pose MLP); "
            "mesh silhouette/seg. Identity PCA frozen after stage 0."
        ),
        train_template_deformer=True,
        lr_template=2e-4,
        **_BOOTSTRAP_RIGID,
        **BOOTSTRAP_TEMPLATE_NO_GP,
    ),
    StageSpec(
        name="2_coarse_mesh",
        steps=15000,
        description=(
            "Global scale/translation + template + expr_mlp + GS (no gamma, no pose MLP); "
            "RGB/sil/lmk/seg; densify."
        ),
        train_template_deformer=True,
        train_expression_deform=True,
        lr_expr_deform=2e-5,
        train_gaussian_appearance=True,
        train_color_pose=False,
        train_color_expression=True,
        train_gaussian_geometry=True,
        train_gaussian_h=True,
        sh_degree=3,
        lr_template=1e-4,
        **_COARSE_RIGID,
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
            "Global rigid frozen; expr_mlp + template + GS detail (no gamma, no pose MLP)."
        ),
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
        **_STAGE3_FROZEN_RIGID,
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
