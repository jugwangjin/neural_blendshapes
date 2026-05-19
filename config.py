"""Training config for MediaPipe → ICT → surface/eye 3DGS stack."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # data
    input_dir: Path = Path("/Bean/data/gwangjin/2024/nbshapes/flare_2/marcel/marcel/marcel")
    train_scenes: list = field(default_factory=lambda: ["MVI_1797", "MVI_1801"])
    eval_scenes: list = field(default_factory=lambda: ["MVI_1802"])
    mp_cache_dir: Path = Path("cache/mediapipe")
    segmentation_dir: Path = Path("cache/segmentation")

    # assets
    ict_npy: Path = Path("assets/ict_facekit_torch.npy")
    mp_embedding: Path = Path(
        "assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz"
    )
    camera_npz: Path = Path("assets/default_camera.npz")

    # model
    num_mp_blendshapes: int = 52
    num_ict_expressions: int = 53
    n_surface_gaussians_per_face: int = 8
    n_surface_gaussians_per_head: int = 8
    n_surface_gaussians_per_mouth_socket: int = 1
    n_surface_gaussians_mouth_interior: int = 4
    gum_h_sigma_scale: float = 4.0
    n_surface_gaussians_per_eye_socket: int = 1
    n_eye_gaussians_per_side: int = 1024
    eye_uv_sample_mode: str = "hemisphere"  # "hemisphere" | "hemisphere_snap" | "triangle"
    eye_sclera_min_front_dot: float = 0.0
    eye_sclera_hemisphere_only: bool = True
    gaussian_scale_knn_k: int = 3
    gaussian_scale_knn_factor: float = 1.0
    n_accessory_gaussians: int = 0
    auto_detect_accessory: bool = False
    accessory_min_pixel_ratio: float = 0.0005
    gaze_uv_range: float = 0.12
    learn_gaze_refine: bool = True
    gamma_min: float = 0.4
    gamma_max: float = 2.5
    au_active_sample_ratio: float = 0.3
    au_active_thresh: float = 0.12

    # train
    use_stage_schedule: bool = True
    batch_size: int = 1
    iterations: int = 60000
    lr_tracker: float = 1e-3
    lr_pose_weight: float = 1e-3
    lr_deformer: float = 1e-3
    lr_gaussian: float = 1e-3
    log_every: int = 50
    save_every: int = 1000
    checkpoint_dir: Path = Path("out/checkpoints")

    # loss weights (defaults; stages override)
    w_rgb: float = 1.0
    w_mp_lmk: float = 1.0
    w_silhouette: float = 10.0
    w_mp_mask: float = 10.0  # alias for silhouette
    w_seg: float = 1.0
    w_iris: float = 1.0
    w_h: float = 0.01
    w_eye_uv_barrier: float = 0.001
    w_scale: float = 0.001
    w_opacity: float = 0.001

    # render (gsplat only)
    image_size: int = 512
    n_semantic_classes: int = 7
    sh_degree: Optional[int] = None

    gsplat_submodule: Path = Path("gsplat")

    # packed=True saves memory; RGB passes with backgrounds fall back to packed=False (gsplat assert)
    gsplat_packed: bool = True