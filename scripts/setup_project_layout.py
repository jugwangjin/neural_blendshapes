"""
One-shot migration: old/flare -> new layout (no FLAME / nvdiffrast / neuralshader).

Run on server:
  python scripts/setup_project_layout.py
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "old" / "flare"

COPY_MAP = [
    ("utils/ict_model.py", "model/ict_model.py"),
    ("modules/neural_blendshapes.py", "model/deformer.py"),
    ("modules/encoder.py", "model/encoder.py"),
    ("modules/resnet.py", "model/resnet.py"),
    ("modules/embedder.py", "model/embedder.py"),
    ("modules/fc.py", "model/fc.py"),
    ("modules/math_np.py", "model/math_np.py"),
    ("dataset/dataset_util.py", "dataset/dataset_util.py"),
    ("dataset/dataset.py", "dataset/collate.py"),
    ("losses/image.py", "losses/rgb.py"),
    ("losses/landmark.py", "losses/mediapipe_landmark.py"),
]

IMPORT_REPLACEMENTS = [
    ("from flare.modules.", "from model."),
    ("from flare.utils.", "from utils."),
    ("from flare.core.", "from utils."),
    ("from flare.losses.", "from losses."),
    ("from flare.dataset.", "from dataset."),
    ("import flare.", "import "),
    ("from .encoder import", "from model.encoder import"),
    ("from flare.modules.embedder import", "from model.embedder import"),
]


def patch_imports(path: Path):
    text = path.read_text(encoding="utf-8")
    for old, new in IMPORT_REPLACEMENTS:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def main():
    dirs = [
        "model",
        "losses",
        "dataset",
        "utils",
        "gaussian_splatting",
        "assets/ict",
        "assets/embeddings",
        "docs/implementation",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    for src_rel, dst_rel in COPY_MAP:
        src = OLD / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        shutil.copy2(src, dst)
        patch_imports(dst)
        print(f"copied {src_rel} -> {dst_rel}")

    mp_src = ROOT / "ict_mediapipe_lmk"
    mp_dst = ROOT / "utils" / "ict_mediapipe_lmk"
    if mp_src.exists() and not mp_dst.exists():
        shutil.copytree(mp_src, mp_dst)
        print(f"copied ict_mediapipe_lmk -> utils/ict_mediapipe_lmk")

    gs_src = ROOT / "gaussian-splatting"
    gs_dst = ROOT / "gaussian_splatting" / "vendor"
    if gs_src.exists() and not gs_dst.exists():
        shutil.copytree(
            gs_src,
            gs_dst,
            ignore=shutil.ignore_patterns("SIBR_viewers", ".git", "__pycache__"),
        )
        print("copied gaussian-splatting -> gaussian_splatting/vendor")

    print("\nDone. New UVH modules (utils/mesh.py, uv_mesh.py, uvh_gaussians.py) are already in repo.")
    print("Next: python train.py  (after wiring config)")


if __name__ == "__main__":
    main()
