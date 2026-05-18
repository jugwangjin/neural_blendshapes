"""Training config (replaces arguments.py + configs_tmp for new stack)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # data
    input_dir: Path = Path("/Bean/data/gwangjin/2024/nbshapes/flare_2/marcel/marcel/marcel")
    train_scenes: list = field(default_factory=lambda: ["MVI_1797", "MVI_1801"])
    eval_scenes: list = field(default_factory=lambda: ["MVI_1802"])

    # assets / processing
    flame_model: Path = Path("processing/flame/FLAME2020/generic_model.pkl")
    ict_npy: Path = Path("assets/ict_facekit_torch.npy")
    mp_embedding: Path = Path(
        "assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz"
    )
    default_camera: Path = Path("assets/default_camera.npz")

    # model
    n_gaussians: int = 65536
    n_eye_gaussians_per_side: int = 64
    gaze_uv_range: float = 0.12
    learn_gaze_refine: bool = True
    ict_canonical: Path = Path("assets/ict_identity.npy")

    # train
    batch_size: int = 1
    iterations: int = 10000
    lr_deformer: float = 1e-3
    lr_gaussian: float = 1e-3

    # loss weights
    w_rgb: float = 1.0
    w_mp_lmk: float = 1.0
    w_mp_mask: float = 1.0
    w_iris: float = 1.0
    w_h_anchor: float = 0.01
    w_eye_uv_barrier: float = 0.001
    w_scale: float = 0.001

    # render
    image_size: int = 512
    gs_root: Path = Path("gaussian_splatting/vendor")
