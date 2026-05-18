"""Paths and indices shared by the ICT MediaPipe landmark baker."""

from pathlib import Path

from processing.paths import (
    ASSETS_DIR,
    FLAME_MODEL,
    FLAME_STATIC_EMBEDDING,
    FLAME_UV_MESH,
    ICT_CANONICAL,
    ICT_NPY,
    LARGE_STEPS_ROOT,
    METRICAL_ROOT,
    PROCESSING_ROOT,
    REPO_ROOT,
)

PACKAGE_ROOT = Path(__file__).resolve().parent

# ICT face patch used for NICP (ICT-FaceKit README)
ICT_FACE_VERTEX_END = 9409
ICT_FACE_FACE_END = 9230

# metrical-tracker/tracker.py (iris vertex indices on FLAME; MP indices 468-477)
LEFT_IRIS_FLAME = [4597, 4542, 4510, 4603, 4570]
RIGHT_IRIS_FLAME = [4051, 3996, 3964, 3932, 4028]
LEFT_IRIS_MP = [468, 469, 470, 471, 472]
RIGHT_IRIS_MP = [473, 474, 475, 476, 477]

# Full-head ICT mesh (24591 verts): iris vertex ranges from ICT-FaceKit README
ICT_LEFT_IRIS_VERTICES = slice(22221, 23020)
ICT_RIGHT_IRIS_VERTICES = slice(23791, 24590)

DEFAULT_METRICAL_ROOT = METRICAL_ROOT
DEFAULT_LARGE_STEPS_ROOT = LARGE_STEPS_ROOT
DEFAULT_MP_EMBEDDING = (
    DEFAULT_METRICAL_ROOT / "flame" / "mediapipe" / "mediapipe_landmark_embedding.npz"
)
DEFAULT_OUTPUT_NPZ = ASSETS_DIR / "ict_mediapipe_landmark_embedding_from_metrical_tracker.npz"
DEFAULT_DEBUG_DIR = PACKAGE_ROOT / "debug"
DEFAULT_FLAME_UV_MESH = FLAME_UV_MESH
DEFAULT_FLAME_MODEL = FLAME_MODEL
DEFAULT_FLAME_LMK_EMBEDDING = FLAME_STATIC_EMBEDDING
DEFAULT_ICT_NPY = ICT_NPY
DEFAULT_ICT_CANONICAL = ICT_CANONICAL
