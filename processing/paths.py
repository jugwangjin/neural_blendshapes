"""Shared repo / processing paths and sys.path bootstrap."""

import sys
from pathlib import Path

PROCESSING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROCESSING_ROOT.parent

FLAME_MODEL = PROCESSING_ROOT / "flame" / "FLAME2020" / "generic_model.pkl"
FLAME_STATIC_EMBEDDING = REPO_ROOT / "assets" / "flame_static_embedding.pkl"
FLAME_UV_MESH = REPO_ROOT / "assets" / "canonical_eye_smpl.obj"
ICT_NPY = REPO_ROOT / "assets" / "ict_facekit_torch.npy"
ICT_CANONICAL = REPO_ROOT / "assets" / "ict_identity.npy"
ASSETS_DIR = REPO_ROOT / "assets"

METRICAL_ROOT = PROCESSING_ROOT / "metrical-tracker"
LARGE_STEPS_ROOT = PROCESSING_ROOT / "large-steps-pytorch"


def setup_import_paths():
    for root in (REPO_ROOT, PROCESSING_ROOT):
        path = str(root)
        if path not in sys.path:
            sys.path.insert(0, path)
