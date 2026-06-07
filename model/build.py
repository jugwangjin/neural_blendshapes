"""Construct ICT stack (tracker, deformer, avatar) from Config."""

from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from model.tracker_mlp import TrackerCorrectionMLP
from utils.sampling import count_surface_gaussians


def sh_dim_from_degree(sh_degree):
    if sh_degree is not None and sh_degree > 0:
        return (sh_degree + 1) ** 2
    return 1


def sh_dim_from_avatar_state(state_dict):
    color_ckpt = state_dict["color"]
    if color_ckpt.ndim == 3:
        return int(color_ckpt.shape[1])
    if color_ckpt.ndim == 2:
        return 1
    raise ValueError(f"unexpected avatar color shape {tuple(color_ckpt.shape)}")


def avatar_checkpoint_layout_kwargs(cfg):
    """Layout / forward flags for ``GaussianAvatar.from_checkpoint_state``."""
    return dict(
        max_scale=cfg.geometry_max_scale,
        with_mesh_scaling=cfg.gaussian_with_mesh_scaling,
        scale_max_clamp_factor=cfg.gaussian_scale_max_clamp_factor,
        expression_support_train_mask=getattr(cfg, "expression_support_train_mask", 0.25),
        color_expression_exclude_mouth_eye=getattr(cfg, "color_expression_exclude_mouth_eye", False),
        n_semantic_classes=getattr(cfg, "n_semantic_classes", 0),
    )


def build_ict(cfg, device):
    return ICTFaceKitTorch(
        npy_dir=str(cfg.ict_npy),
        mouth_interior_jaw_only_expression=cfg.mouth_interior_jaw_only_expression,
    ).to(device)


def build_tracker(cfg, ict, device):
    return TrackerCorrectionMLP(
        mediapipe_to_ict=ict.mediapipe_to_ict,
        num_ict_expression=ict.num_expression,
        n_blendshapes=cfg.num_mp_blendshapes,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
        gamma_symmetric_log=cfg.gamma_symmetric_log,
        use_landmarks=True,
        additive_gamma_correction=getattr(cfg, "additive_gamma_correction", False),
    ).to(device)


def build_deformer(cfg, ict, device):
    expr_region_weight = build_expr_region_weight(ict).to(device)
    return ICTDeformer(
        ict,
        expr_region_weight,
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
        n_coeffs=cfg.num_ict_expressions,
        mouth_interior_jaw_only_expression=cfg.mouth_interior_jaw_only_expression,
    ).to(device)


def build_avatar(cfg, ict, deformer, device):
    sh_degree = getattr(cfg, "sh_degree", None)
    sh_dim = sh_dim_from_degree(sh_degree)
    return GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        sh_dim=sh_dim,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_teeth=cfg.n_surface_gaussians_per_teeth,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        n_semantic_classes=cfg.n_semantic_classes,
        gaussian_scale_knn_k=cfg.gaussian_scale_knn_k,
        gaussian_scale_knn_factor=cfg.gaussian_scale_knn_factor,
        face_center_init=cfg.gaussian_face_center_init,
        max_scale=cfg.geometry_max_scale,
        with_mesh_scaling=cfg.gaussian_with_mesh_scaling,
        scale_max_clamp_factor=cfg.gaussian_scale_max_clamp_factor,
        expression_support_train_mask=cfg.expression_support_train_mask,
        color_expression_exclude_mouth_eye=cfg.color_expression_exclude_mouth_eye,
    ).to(device)


def estimate_init_gaussian_count(cfg, ict) -> int:
    """Surface Gaussians at ``from_ict`` layout (before densify)."""
    return count_surface_gaussians(
        ict,
        ict.faces,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_teeth=cfg.n_surface_gaussians_per_teeth,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        k_face_loose_factor=1.0,
        face_center_init=cfg.gaussian_face_center_init,
    )


def print_surface_gaussian_count(cfg, ict):
    n_surface = count_surface_gaussians(
        ict,
        ict.faces,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_teeth=cfg.n_surface_gaussians_per_teeth,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        k_face_loose_factor=1.0,
        face_center_init=cfg.gaussian_face_center_init,
    )
    print(
        f"surface Gaussians: {n_surface} "
        f"(face/head={cfg.n_surface_gaussians_per_face}/{cfg.n_surface_gaussians_per_head} "
        f"mouth_socket={cfg.n_surface_gaussians_per_mouth_socket} "
        f"mouth_interior={cfg.n_surface_gaussians_mouth_interior} "
        f"teeth={cfg.n_surface_gaussians_per_teeth} "
        f"eye_socket={cfg.n_surface_gaussians_per_eye_socket} "
        f"sclera/occlusion={cfg.n_surface_gaussians_per_eyeball_sclera}/"
        f"{cfg.n_surface_gaussians_per_eye_occlusion} per face; "
        "h reg: FLARE image masks only)"
    )
