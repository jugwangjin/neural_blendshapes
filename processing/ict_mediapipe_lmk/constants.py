"""Paths and indices for ICT MediaPipe landmark baker."""

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

# NICP face patch: part #0 skin verts; faces = tris with all corners in [0, VERT_END)
ICT_FACE_VERTEX_END = 9409
ICT_FACE_FACE_END = 9230  # legacy README count; actual F = filter by vertex_end

# metrical-tracker/tracker.py — iris on FLAME; MP indices 468–477
LEFT_IRIS_FLAME = [4597, 4542, 4510, 4603, 4570]
RIGHT_IRIS_FLAME = [4051, 3996, 3964, 3932, 4028]
LEFT_IRIS_MP = [468, 469, 470, 471, 472]
RIGHT_IRIS_MP = [473, 474, 475, 476, 477]

DEFAULT_METRICAL_ROOT = METRICAL_ROOT
DEFAULT_LARGE_STEPS_ROOT = LARGE_STEPS_ROOT
DEFAULT_MP_EMBEDDING = (
    DEFAULT_METRICAL_ROOT / "flame" / "mediapipe" / "mediapipe_landmark_embedding.npz"
)
# Minimal train asset (mp id + ict_lmk_face_idx + ict_lmk_b_coords)
DEFAULT_OUTPUT_NPZ = ASSETS_DIR / "ict_mediapipe_landmark_indices.npz"
DEFAULT_OUTPUT_LEGACY = ASSETS_DIR / "ict_mediapipe_landmark_embedding_from_metrical_tracker.npz"
DEFAULT_DEBUG_DIR = PACKAGE_ROOT / "debug"
DEFAULT_FLAME_UV_MESH = FLAME_UV_MESH
DEFAULT_FLAME_MODEL = FLAME_MODEL
DEFAULT_FLAME_LMK_EMBEDDING = FLAME_STATIC_EMBEDDING
DEFAULT_ICT_NPY = ICT_NPY
DEFAULT_ICT_CANONICAL = ICT_CANONICAL
