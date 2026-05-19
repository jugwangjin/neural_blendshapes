"""Load MediaPipe landmark embedding from metrical-tracker."""

import pickle

import numpy as np

from processing.ict_mediapipe_lmk.constants import (
    LEFT_IRIS_FLAME,
    LEFT_IRIS_MP,
    RIGHT_IRIS_FLAME,
    RIGHT_IRIS_MP,
)
from processing.ict_mediapipe_lmk.landmarks import sample_bary


def load_flame_static_embedding(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return (
        data["lmk_face_idx"].astype(np.int64),
        data["lmk_b_coords"].astype(np.float64),
    )


def load_metrical_mediapipe_embedding(path):
    mp_emb = np.load(path, allow_pickle=True, encoding="latin1")
    return (
        mp_emb["landmark_indices"].astype(np.int64),
        mp_emb["lmk_face_idx"].astype(np.int64),
        mp_emb["lmk_b_coords"].astype(np.float64),
    )


def sample_flame_mediapipe_landmarks(v_flame, f_flame, mp_embedding_path):
    mp_ids, flame_face_idx, flame_bary = load_metrical_mediapipe_embedding(mp_embedding_path)
    p_mp = sample_bary(v_flame, f_flame, flame_face_idx, flame_bary)
    return mp_ids, p_mp


def sample_flame_iris_landmarks(v_flame):
    left_iris_flame = np.array(LEFT_IRIS_FLAME, dtype=np.int64)
    right_iris_flame = np.array(RIGHT_IRIS_FLAME, dtype=np.int64)
    left_iris_mp = np.array(LEFT_IRIS_MP, dtype=np.int64)
    right_iris_mp = np.array(RIGHT_IRIS_MP, dtype=np.int64)

    iris_mp_ids = np.concatenate([left_iris_mp, right_iris_mp])
    p_iris = v_flame[np.concatenate([left_iris_flame, right_iris_flame])]
    target_type = ["left_iris"] * len(left_iris_mp) + ["right_iris"] * len(right_iris_mp)
    return left_iris_mp, right_iris_mp, iris_mp_ids, p_iris, target_type


def build_flame_mp_points(v_flame, f_flame, mp_embedding_path):
    mp_ids, p_mp = sample_flame_mediapipe_landmarks(v_flame, f_flame, mp_embedding_path)
    left_iris_mp, right_iris_mp, iris_mp_ids, p_iris, iris_types = sample_flame_iris_landmarks(
        v_flame
    )

    all_mp_ids = np.concatenate([mp_ids, iris_mp_ids])
    all_p_flame = np.concatenate([p_mp, p_iris], axis=0)
    target_type = ["face"] * len(mp_ids) + iris_types
    source = (
        ["metrical-tracker"] * len(mp_ids)
        + ["iris_hardcoded"] * len(iris_mp_ids)
    )

    return {
        "mp_ids": all_mp_ids,
        "points_flame": all_p_flame,
        "target_type": target_type,
        "source": source,
        "left_iris_mp": left_iris_mp,
        "right_iris_mp": right_iris_mp,
        "skin_mp_ids": mp_ids,
        "skin_points_flame": p_mp,
    }
