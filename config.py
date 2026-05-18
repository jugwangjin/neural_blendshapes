"""Training config for MediaPipe → ICT → UVH/3DGS stack (no FLAME/DECA at runtime)."""

from dataclasses import dataclass, field
from pathlib import Path


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
    ict_canonical: Path = Path("assets/ict_identity.npy")
    mp_embedding: Path = Path(
        "assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz"
    )
    camera_npz: Path = Path("assets/default_camera.npz")

    # model
    num_mp_blendshapes: int = 52
    num_ict_expressions: int = 53
    n_face_gaussians: int = 4096
    n_eye_gaussians_per_side: int = 64
    gaze_uv_range: float = 0.12
    learn_gaze_refine: bool = True
    gamma_min: float = 0.4
    gamma_max: float = 2.5
    au_active_sample_ratio: float = 0.3
    au_active_thresh: float = 0.12

    # train (iterations overridden by training.stages.STAGE_SCHEDULE in train.py)
    use_stage_schedule: bool = True
    stage: str = "legacy"
    batch_size: int = 1
    iterations: int = 55000
    lr_tracker: float = 1e-3
    lr_pose_weight: float = 1e-3
    lr_deformer: float = 1e-3
    lr_gaussian: float = 1e-3
    log_every: int = 50
    save_every: int = 1000
    checkpoint_dir: Path = Path("out/checkpoints")

    # loss weights
    w_rgb: float = 1.0
    w_mp_lmk: float = 1.0
    w_mp_mask: float = 1.0
    w_seg: float = 1.0
    w_iris: float = 1.0
    w_h: float = 0.01
    w_eye_uv_barrier: float = 0.001
    w_scale: float = 0.001
    w_opacity: float = 0.001

    # render (gsplat only — see rendering/avatar_renderer.py)
    image_size: int = 512
    n_semantic_classes: int = 7

    # gsplat: pip install gsplat OR git submodule at ./gsplat (do not edit submodule)
    gsplat_submodule: Path = Path("gsplat")
