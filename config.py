"""Training config for MediaPipe → ICT → surface/eye 3DGS stack."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

SplitNames = Union[str, list[str]]


@dataclass
class Config:
    # data — default ``ImageDataset`` (``dataset/image_dataset.py``)
    # ``input_dir/{scene}/image`` — scene name or list of scenes (e.g. ``["MVI_1797", "MVI_1801"]``)
    dataset_type: str = "flare"  # ``flare``|``image`` → ImageDataset layout; ``mp_npz`` → VideoDataset
    input_dir: Path = Path("/Bean/data/gwangjin/2024/nbshapes/flare_2/justin/justin")
    train_split: SplitNames = "train"
    eval_split: SplitNames = "test"
    mp_cache_dir: Path = Path("cache/mediapipe")
    mask_edt_cache_dir: Path = Path("cache/mask_edt")
    rebuild_mask_edt_cache: bool = False
    segmentation_dir: Path = Path("cache/segmentation")
    face_landmarker_task: Path = Path("assets/face_landmarker.task")
    rebuild_mp_cache: bool = False
    bshapes_mode_percentile: float = 10.0
    fa_bbox_scale: float = 1.25
    distribution_sample_ratio: float = 0.35
    distribution_low_weight: float = 0.05
    distribution_high_cap: float = 1.0
    distribution_var_eps: float = 5e-2
    eye_blink_median_target: float = 0.9  # Deprecated
    eye_blink_lo_percentile: float = 10.0

    eye_au_min_range: float = 0.4

    # assets
    ict_npy: Path = Path("assets/ict_facekit_torch.npy")
    mediapipe_name_to_ict: Path = Path("assets/mediapipe_name_to_indices.pkl")
    mp_embedding: Path = Path("assets/ict_mediapipe_landmark_indices.npz")
    camera_npz: Path = Path("assets/default_camera.npz")

    # model
    num_mp_blendshapes: int = 52
    num_ict_expressions: int = 53
    n_surface_gaussians_per_face: int = 8
    n_surface_gaussians_per_head: int = 8
    n_surface_gaussians_per_mouth_socket: int = 1
    n_surface_gaussians_mouth_interior: int = 2
    gum_h_sigma_scale: float = 4.0
    n_surface_gaussians_per_eye_socket: int = 1
    n_surface_gaussians_per_eyeball_sclera: int = 0
    n_surface_gaussians_per_eye_occlusion: int = 8
    gaussian_scale_knn_k: int = 4
    gaussian_scale_knn_factor: float = 0.5
    gamma_min: float = 0.4
    gamma_max: float = 2.5
    au_active_sample_ratio: float = 0.3  # legacy ``mp_npz`` VideoDataset only
    au_active_thresh: float = 0.12

    # train
    use_stage_schedule: bool = True
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    iterations: int = 60000
    lr_tracker: float = 1e-3
    lr_pose_weight: float = 1e-3
    lr_deformer: float = 1e-3
    lr_gaussian: float = 1e-3
    log_every: int = 50
    grad_clip_max_norm: float = 5.0
    output_root: Path = Path("/Bean/log/gwangjin/2026/neural_blendshapes_2/justin")
    eval_max_frames: int = 500  # 0 = all eval frames per stage-end render

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def codes_dir(self) -> Path:
        return self.output_root / "codes"

    @property
    def eval_render_dir(self) -> Path:
        return self.output_root / "renders"

    # loss weights (defaults; stages override)
    w_rgb: float = 1.0
    w_mp_lmk: float = 1.0
    mp_lmk_iris_weight: float = 2.5  # per-point multiplier for MP iris 468–477 inside mp_lmk loss
    w_pie68_jaw: float = 0.0  # PIE 68 jawline 0:16 (FA detections; MP has no chin contour)
    pie68_jaw_score_thresh: float = 0.3
    w_silhouette: float = 10.0
    w_mp_mask: float = 10.0  # alias for silhouette
    silhouette_use_edt: bool = False  # default L2; set True to use precomputed GT EDT loss
    silhouette_edt_w_ext: float = 1.0
    silhouette_edt_w_int: float = 1.0
    silhouette_edt_max_dist_px: float = 50.0
    precompute_mask_edt_cache: bool = False  # build cache/mask_edt on dataset init
    mask_edt_downsample: int = 4
    mask_edt_normalize: bool = False  # keep pixel EDT; loss applies clamp / max_dist_px
    w_seg: float = 0.0
    w_h: float = 0.01
    w_geometry: float = 0.01
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
    opacity_target: float = 1.0
    opacity_w_skin: float = 1.0
    opacity_w_other: float = 0.05
    w_pose_prior: float = 0.0
    w_template_smooth: float = 0.0
    w_opacity_decay: float = 0.0

    # render (gsplat only)
    image_size: int = 512
    n_semantic_classes: int = 0
    sh_degree: Optional[int] = None
    gsplat_rasterize_mode: str = "classic"

    gaussian_face_center_init: bool = True
    gaussian_densify: bool = True
    gaussian_densify_max: int = 200000
    gaussian_densify_stages: list[str] = field(default_factory=lambda: ["2_coarse_mesh"])
    gaussian_densify_start_iter: int = 5000
    gaussian_densify_stop_iter: int = 15000
    gaussian_densify_every: int = 100
    gaussian_densify_accum_every: int = 1
    gaussian_prune_every: int = 100
    gaussian_reset_every: int = 3000
    gaussian_grow_grad2d: float = 0.0002
    gaussian_grow_gradrgb: float = 0.0000005
    gaussian_grow_option: str = "grad2d"
    gaussian_prune_opa: float = 0.005
    gaussian_triangle_walk_every: int = 10

    gsplat_submodule: Path = Path("gsplat")

    # packed=True saves memory but gsplat packed backward can yield NaN grads (see debug_nan.py).
    gsplat_packed: bool = False