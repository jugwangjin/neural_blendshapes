"""
MediaPipe cache [B, 52] → ICT FaceKit expression coeffs [B, 53].

``mediapipe_name_to_indices.pkl``: category name → column in ``mp_blendshape``.

``load_mediapipe_mapping(pkl)`` reads once and returns indices + gather table + cache labels + eye columns.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

NUM_MP_BLENDSHAPE_CHANNELS = 52
NUM_MEDIAPIPE_TO_ICT = 53

DEFAULT_MEDIAPIPE_NAME_TO_ICT = (
    Path(__file__).resolve().parent.parent / "assets/mediapipe_name_to_indices.pkl"
)

# ICTFaceKitTorch.load_mediapipe_idx — gather slot order (53 MediaPipe category names)
ICT_GATHER_MP_NAMES = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)

LEFT_IRIS_MP = [468, 469, 470, 471, 472]
RIGHT_IRIS_MP = [473, 474, 475, 476, 477]
IRIS_MP = LEFT_IRIS_MP + RIGHT_IRIS_MP


@dataclass
class MediapipeMapping:
    name_to_idx: dict
    mediapipe_to_ict: torch.Tensor
    cache_channel_names: list
    eye_blink_l: int
    eye_blink_r: int
    eye_wide_l: int
    eye_wide_r: int
    eye_look_down_l: int
    eye_look_down_r: int
    eye_look_in_l: int
    eye_look_in_r: int
    eye_look_out_l: int
    eye_look_out_r: int
    eye_look_up_l: int
    eye_look_up_r: int

    def __iter__(self):
        yield self.name_to_idx
        yield self.mediapipe_to_ict


_default_mapping = None


def default_mediapipe_mapping(pkl_path=None):
    global _default_mapping
    if _default_mapping is None:
        path = DEFAULT_MEDIAPIPE_NAME_TO_ICT if pkl_path is None else pkl_path
        _default_mapping = load_mediapipe_mapping(path)
    return _default_mapping


def load_mediapipe_mapping(pkl_path, num_expression=None):
    with open(pkl_path, "rb") as f:
        name_to_idx = pickle.load(f)

    missing = [n for n in ICT_GATHER_MP_NAMES if n not in name_to_idx]
    if missing:
        raise KeyError(f"{pkl_path}: missing keys (first 5): {missing[:5]}")

    mediapipe_to_ict = torch.tensor(
        [int(name_to_idx[n]) for n in ICT_GATHER_MP_NAMES], dtype=torch.long
    )
    if mediapipe_to_ict.numel() != NUM_MEDIAPIPE_TO_ICT:
        raise ValueError(
            f"mediapipe_to_ict length {mediapipe_to_ict.numel()} != {NUM_MEDIAPIPE_TO_ICT}"
        )
    lo = int(mediapipe_to_ict.min().item())
    hi = int(mediapipe_to_ict.max().item())
    if lo < 0 or hi >= NUM_MP_BLENDSHAPE_CHANNELS:
        raise ValueError(
            f"mediapipe_to_ict column index [{lo}, {hi}] out of range for width "
            f"{NUM_MP_BLENDSHAPE_CHANNELS}"
        )
    if num_expression is not None and mediapipe_to_ict.numel() != int(num_expression):
        raise ValueError(
            f"mediapipe_to_ict length {mediapipe_to_ict.numel()} != num_expression={num_expression}"
        )

    m = name_to_idx
    return MediapipeMapping(
        name_to_idx=name_to_idx,
        mediapipe_to_ict=mediapipe_to_ict,
        cache_channel_names=_cache_channel_names(name_to_idx),
        eye_blink_l=int(m["eyeBlinkLeft"]),
        eye_blink_r=int(m["eyeBlinkRight"]),
        eye_wide_l=int(m["eyeWideLeft"]),
        eye_wide_r=int(m["eyeWideRight"]),
        eye_look_down_l=int(m["eyeLookDownLeft"]),
        eye_look_down_r=int(m["eyeLookDownRight"]),
        eye_look_in_l=int(m["eyeLookInLeft"]),
        eye_look_in_r=int(m["eyeLookInRight"]),
        eye_look_out_l=int(m["eyeLookOutLeft"]),
        eye_look_out_r=int(m["eyeLookOutRight"]),
        eye_look_up_l=int(m["eyeLookUpLeft"]),
        eye_look_up_r=int(m["eyeLookUpRight"]),
    )


def _cache_channel_names(name_to_idx):
    names = [""] * NUM_MP_BLENDSHAPE_CHANNELS
    for name, idx in name_to_idx.items():
        j = int(idx)
        if names[j]:
            names[j] = f"{names[j]}__{name}"
        else:
            names[j] = name
    for j in range(NUM_MP_BLENDSHAPE_CHANNELS):
        if not names[j]:
            names[j] = f"mp_col_{j}"
    return names


def mp_blendshape_name_aliases(name: str) -> tuple[str, ...]:
    """
    ARKit-style ``*Left`` / ``*Right`` → ICT pkl keys (``*_L`` / ``*_R`` or merged stem).

    ``mediapipe_name_to_indices.pkl`` follows ICT gather names (e.g. ``browInnerUp`` twice),
    not always split ``browInnerUpLeft``.
    """
    if name in ICT_GATHER_MP_NAMES:
        return ()
    if name.endswith("Left"):
        stem = name[:-4]
        return (f"{stem}_L", stem)
    if name.endswith("Right"):
        stem = name[:-5]
        return (f"{stem}_R", stem)
    return ()


def resolve_mp_blendshape_index(name: str, name_to_idx: dict[str, int]) -> int:
    """Map control-sequence / ARKit name → column in ``mp_blendshape`` cache."""
    if name in name_to_idx:
        return int(name_to_idx[name])
    for alt in mp_blendshape_name_aliases(name):
        if alt in name_to_idx:
            return int(name_to_idx[alt])
    tried = (name, *mp_blendshape_name_aliases(name))
    raise KeyError(f"unknown MediaPipe blendshape {name!r} (tried {list(tried)})")


def mp_to_ict_expression_weights(mp_coeffs, mediapipe_to_ict, num_expression=None):
    if mp_coeffs.shape[-1] != NUM_MP_BLENDSHAPE_CHANNELS:
        raise ValueError(
            f"mp_coeffs last dim {mp_coeffs.shape[-1]} != {NUM_MP_BLENDSHAPE_CHANNELS}"
        )
    idx = mediapipe_to_ict.to(device=mp_coeffs.device, dtype=torch.long).reshape(-1)
    out = mp_coeffs[:, idx]
    if num_expression is not None and out.shape[-1] != int(num_expression):
        raise ValueError(
            f"gathered shape {out.shape[-1]} != num_expression={num_expression}"
        )
    return out


def __getattr__(name):
    legacy = {
        "MP_EYE_LOOK_DOWN_L": "eye_look_down_l",
        "MP_EYE_LOOK_DOWN_R": "eye_look_down_r",
        "MP_EYE_LOOK_IN_L": "eye_look_in_l",
        "MP_EYE_LOOK_IN_R": "eye_look_in_r",
        "MP_EYE_LOOK_OUT_L": "eye_look_out_l",
        "MP_EYE_LOOK_OUT_R": "eye_look_out_r",
        "MP_EYE_LOOK_UP_L": "eye_look_up_l",
        "MP_EYE_LOOK_UP_R": "eye_look_up_r",
        "MP_EYE_BLINK_L": "eye_blink_l",
        "MP_EYE_BLINK_R": "eye_blink_r",
        "MP_EYE_WIDE_L": "eye_wide_l",
        "MP_EYE_WIDE_R": "eye_wide_r",
    }
    if name in legacy:
        return getattr(default_mediapipe_mapping(), legacy[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
