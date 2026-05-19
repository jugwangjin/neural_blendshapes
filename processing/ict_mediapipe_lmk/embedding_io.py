"""Save/load minimal ICT MediaPipe barycentric embedding (train + texture viz)."""

from pathlib import Path

import numpy as np

from processing.ict_mediapipe_lmk.constants import (
    DEFAULT_OUTPUT_LEGACY,
    DEFAULT_OUTPUT_NPZ,
)

MINIMAL_KEYS = ("mp_landmark_indices", "ict_lmk_face_idx", "ict_lmk_b_coords")


def assert_flame_mp_embedding(mp_embedding_path, n_flame_faces):
    z = np.load(mp_embedding_path, allow_pickle=True)
    key = "lmk_face_idx" if "lmk_face_idx" in z else "flame_face_idx"
    mx = int(np.asarray(z[key]).max())
    if mx >= n_flame_faces:
        raise ValueError(
            f"FLAME MP embedding max face index {mx} >= F={n_flame_faces} ({mp_embedding_path}). "
            "Use matching topology (pkl: default; processed: --use_processed_faces)."
        )


def save_ict_mediapipe_embedding(path, embedding):
    """Train/viz asset: MP id + ICT face index + barycentric weights only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mp_landmark_indices=np.asarray(embedding["mp_landmark_indices"], dtype=np.int64),
        ict_lmk_face_idx=np.asarray(embedding["ict_lmk_face_idx"], dtype=np.int64),
        ict_lmk_b_coords=np.asarray(embedding["ict_lmk_b_coords"], dtype=np.float32),
    )


def save_ict_mediapipe_embedding_aux(debug_path, embedding, v_ict_fit, f_ict, regions):
    """Optional debug bundle (not needed for train)."""
    debug_path = Path(debug_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        debug_path,
        transfer_error=embedding["transfer_error"],
        ict_lmk_target_type=embedding["ict_lmk_target_type"],
        geometry_chart_id=embedding["geometry_chart_id"],
        source=embedding["source"],
        v_ict_fit=np.asarray(v_ict_fit, dtype=np.float32),
        ict_faces=np.asarray(f_ict, dtype=np.int64),
        ict_asset_variant=regions["asset_variant"],
        ict_asset_schema_version=regions["asset_schema_version"],
    )


def load_ict_mediapipe_embedding(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in MINIMAL_KEYS if k in d}


def resolve_embedding_path(path=None):
    """Prefer minimal asset; fall back to legacy long filename."""
    if path:
        return Path(path)
    for p in (DEFAULT_OUTPUT_NPZ, DEFAULT_OUTPUT_LEGACY):
        if p.is_file():
            return p
    return DEFAULT_OUTPUT_NPZ
