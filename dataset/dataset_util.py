# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

from pathlib import Path

import torch
import imageio
import numpy as np
import cv2
import skimage
from tqdm import tqdm

import mediapipe as mp

from utils.frame_sort import sort_frame_paths
from utils.mediapipe_blendshapes import MP_EYE_BLINK_L, MP_EYE_BLINK_R, MP_EYE_WIDE_L, MP_EYE_WIDE_R

def parse_mediapipe_output(face_landmarker_result):
    if len(face_landmarker_result.face_landmarks) == 0:
        return None, None, None
    landmarks = face_landmarker_result.face_landmarks[0]
    lmks = torch.from_numpy(np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))]).astype(np.float32))

    blendshapes = face_landmarker_result.face_blendshapes[0]
    # print(blendshapes[0])
    bshape = torch.from_numpy(np.array([blendshapes[i].score for i in range(len(blendshapes))]).astype(np.float32))
    
    transform_matrix = torch.from_numpy(face_landmarker_result.facial_transformation_matrixes[0].astype(np.float32))

    return lmks, bshape, transform_matrix
    
###############################################################################
# Helpers/utils
###############################################################################

def _load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def _load_mask(fn):
    alpha = imageio.imread(fn, mode='F') 
    alpha = skimage.img_as_float32(alpha)
    mask = torch.tensor(alpha / 255., dtype=torch.float32).unsqueeze(-1)
    mask[mask < 0.5] = 0.0
    # alpha = imageio.imread(fn) 
    # mask = torch.Tensor(np.array(alpha) > 127.5)[:, :, 1:2].bool().int().float()
    return mask

def _load_img(fn):
    img = imageio.imread(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        # look into this
        img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _load_semantic(fn):
    img = imageio.imread(fn, mode='F')
    h, w = img.shape
    semantics = np.zeros((h, w, 8))
    # Labels that ICT have
    # face, head/neck/, left eye, right eye, mouth interior
    # face + eyebrow + nose + upper lip + lower lip + ears +  == ICT-FaceKit.full_face_area
    # left eye == ICT_FaceKit.eyeball_left
    # right eye == ICT_FaceKit.eyeball_right
    # mouth interior == ICT_FaceKit.mouth_interior == ICT_Facekit.outh_socket + ICT_Facekit.gums_and_tongue + ICT_FaceKit.teeth
    # hair + cloth + necklace + neck == ICT_FaceKit.head_and_neck
    # What I missed
    # part_idx = {
    #     'background': 0,
    #     'skin': 1,
    #     'l_brow': 2,
    #     'r_brow': 3,
    #     'l_eye': 4,
    #     'r_eye': 5,
    #     'eye_g': 6, # eyeglasses, ignored
    #     'l_ear': 7,
    #     'r_ear': 8,
    #     'ear_r': 9,
    #     'nose': 10,
    #     'mouth': 11,
    #     'u_lip': 12,
    #     'l_lip': 13,
    #     'neck': 14,
    #     'neck_l': 15, # necklace
    #     'cloth': 16,
    #     'hair': 17,
    #     'hat': 18
    # }


    # skin, eyebrows, nose, lips, neck, ear, hair, cloth, hat, glasses
    semantics[:, :, 0] = ((img == 1) + (img == 2) + (img == 3) + (img == 10) + (img == 12) + (img == 13)\
                        + (img == 17) + (img == 16) + (img == 15) + (img == 14) + (img==7) + (img==8) + (img==9)\
                            ) >= 1 

    # skin, eyebrows, nose, lips
    semantics[:, :, 1] = ((img == 1) + (img == 2) + (img == 3) + (img == 10) + (img == 12) + (img == 13)
                        ) >= 1 


    # skin, ears, nose, neck
    semantics[:, :, 2] = ((img == 1) + (img == 7) + (img == 8) + (img == 10) + (img == 14)) >= 1

    # left eyes
    semantics[:, :, 3] = ((img == 5)) >= 1

    # right eyes
    semantics[:, :, 4] = ((img == 4)) >= 1

    # inside mouth
    semantics[:, :, 5] = (img == 11) >= 1  # will it include the teeth and 

    # skin, eyebrows, nose, lips -> tight face
    semantics[:, :, 6] = ((img == 1) + (img == 2) + (img == 3) + (img == 10) + (img == 12) + (img == 13) \
                        ) >= 1 
    
    semantics[:, :, 7] = 1. - np.sum(semantics[:, :, :-1], 2) # background

    semantics = torch.tensor(semantics, dtype=torch.float32)
    return semantics


#----------------------------------------------------------------------------
# sRGB color transforms:Code adapted from Nvdiffrec
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


# -----------------------------------------------------------------------------
# FLARE layout paths
# -----------------------------------------------------------------------------

def normalize_split_names(split) -> list[str]:
    """``"train"`` | ``["MVI_1797", "MVI_1801"]`` → list of scene folder names."""
    if split is None:
        return []
    if isinstance(split, str):
        return [split]
    return [str(s) for s in split]


def format_splits_label(split) -> str:
    names = normalize_split_names(split)
    if len(names) == 0:
        return ""
    if len(names) == 1:
        return names[0]
    return "+".join(names)


def scene_tag_from_image(subject_root: Path, img_path: Path) -> str:
    """``{subject_root}/{scene}/image/<stem>.png`` → ``scene``."""
    rel = Path(img_path).resolve().relative_to(Path(subject_root).resolve())
    return rel.parts[0]


def list_split_images(subject_root: Path, split):
    """Numeric-sort ``.png`` merged from ``subject_root / {scene} / image`` for each scene in ``split``."""
    subject_root = Path(subject_root)
    out = []
    for scene in normalize_split_names(split):
        img_dir = subject_root / scene / "image"
        if img_dir.is_dir():
            out.extend(sort_frame_paths(list(img_dir.glob("*.png"))))
    return out


def paths_for_image(img_path: Path):
    """``.../split/image/<stem>.png`` → mask / semantic / seg_mask / normal siblings."""
    img_path = Path(img_path)
    split_root = img_path.parent.parent
    stem = img_path.stem
    return {
        "image": img_path,
        "mask": split_root / "mask" / f"{stem}.png",
        "seg_mask": split_root / "seg_mask" / f"{stem}.png",
        "normal": split_root / "normal" / f"{stem}.png",
        "semantic": split_root / "semantic" / f"{stem}.png",
        "semantic_color": split_root / "semantic_color" / f"{stem}.png",
    }


def load_gt_normal(path: Path, image_size: int, mask=None):
    """
    Face normal PNG from ``prepare_normals`` → ``[3,H,W]`` unit vectors in [-1, 1].

    Background (outside mask if given) is zero.
    """
    import cv2

    path = Path(path)
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.shape[0] != image_size or arr.shape[1] != image_size:
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    chw = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
    chw = chw * 2.0 - 1.0
    nrm = torch.linalg.norm(chw, dim=0, keepdim=True).clamp(min=1e-8)
    chw = chw / nrm
    if mask is not None:
        m = mask
        if m.ndim == 3:
            m = m[0]
        chw = chw * m
    return chw


# -----------------------------------------------------------------------------
# Blendshape mode + AU/pose sampling weights
# -----------------------------------------------------------------------------

def matrix_to_pose_feat(T: torch.Tensor) -> torch.Tensor:
    """4×4 MP facial transform → [6 rot6d | 3 translation] (11-D)."""
    from utils.mesh_ops import rotation_matrix_to_6d

    R = T[:3, :3].float()
    t = T[:3, 3].float()
    scale = R.norm(dim=0).mean().clamp(min=1e-8)
    R = R / scale
    r6 = rotation_matrix_to_6d(R)
    return torch.cat([r6, t], dim=0)


def compute_bshapes_mode(blendshapes_stack: np.ndarray, percentile: float = 10.0) -> torch.Tensor:
    """Per-AU percentile (FLARE: 10th) — resting face; eyeBlink low ≈ eyes open."""
    percentiles = [
        float(np.percentile(blendshapes_stack[:, au], percentile))
        for au in range(blendshapes_stack.shape[1])
    ]
    return torch.tensor(percentiles, dtype=torch.float32)


def compute_distribution_weights(
    blendshapes: np.ndarray,
    pose_feats: np.ndarray,
    bshapes_mode: torch.Tensor = None,
    *,
    coeffs_are_calibrated: bool = False,
    var_eps: float = 5e-2,
    low_weight: float = 0.05,
    high_cap: float = 1.0,
) -> np.ndarray:
    """Near mean → low weight; outliers → high (FLARE importance-style).

    Feature = MP coeffs (52, calibrated or mode-centered) + pose rotation 6D.
    Per-dimension variance on the train split (diagonal Mahalanobis):
    ``sqrt(sum_j (feat_c_j^2 / (var_j + var_eps)))``.
    """
    pose_rot = pose_feats[:, :6]
    if coeffs_are_calibrated:
        bs_feat = blendshapes.astype(np.float64)
    else:
        bs_feat = (blendshapes - bshapes_mode.cpu().numpy()[None, :]).astype(np.float64)
    feat = np.concatenate([bs_feat, pose_rot.astype(np.float64)], axis=1)
    feat_c = feat - feat.mean(axis=0, keepdims=True)
    n = max(feat_c.shape[0] - 1, 1)
    var = np.sum(feat_c ** 2, axis=0) / n + float(var_eps)
    score = np.sqrt(np.sum(feat_c ** 2 / var[None, :], axis=1))
    score = score / (np.amax(score) / 2.0 + 1e-8)
    return np.clip(score, low_weight, high_cap).astype(np.float32)


def merge_distribution_with_rgb_ema(
    pca_weights: np.ndarray,
    rgb_ema: np.ndarray,
    *,
    ema_scale: float = 1.0,
    ema_max_lift: float = 0.5,
    low_weight: float = 0.05,
    high_cap: float = 1.0,
) -> np.ndarray:
    """
    Lift sample weight toward the MP+pose distribution maximum, never above it.

    MP+pose variance weights stay the baseline; RGB EMA only closes part of the
    gap to ``max(dist)``. No global re-normalization (EMA must not reshape the whole distribution).
    """
    pca = np.asarray(pca_weights, dtype=np.float64)
    ema = np.asarray(rgb_ema, dtype=np.float64)
    if ema_scale <= 0.0 or ema_max_lift <= 0.0 or ema.max() <= 1e-8:
        return pca.astype(np.float32)
    pca_max = float(np.max(pca))
    ema_n = ema / (ema.max() + 1e-8)
    lift = np.clip(float(ema_scale) * ema_n, 0.0, float(ema_max_lift))
    merged = pca + lift * (pca_max - pca)
    merged = np.minimum(merged, pca_max)
    return np.clip(merged, low_weight, high_cap).astype(np.float32)


def compute_eye_au_calibration(
    blendshapes_stack: np.ndarray,
    bshapes_mode: torch.Tensor,
    *,
    blink_lo_percentile: float = 10.0,  # 눈을 뜬 상태 (작은 눈 베이스라인 커버)
    min_range: float = 0.4,             # 블링크 변화량의 최소 보장폭
):
    """
    Eye Blink 캘리브레이션: 하위 백분위수를 0으로, 시퀀스 max를 1.0으로 스케일링.
    eyeWide: 일반 FAU로 처리되므로 (raw - mode).clamp(0,1) 사용.
    """
    lo_list = []
    hi_list = []
    for j in (MP_EYE_BLINK_L, MP_EYE_BLINK_R):
        col = blendshapes_stack[:, j]
        lo_v = float(np.percentile(col, blink_lo_percentile))
        hi_v = float(np.max(col))
        span = hi_v - lo_v
        if span < min_range:
            hi_v = lo_v + min_range
        lo_list.append(lo_v)
        hi_list.append(hi_v)
    return {
        "blink_idx": [MP_EYE_BLINK_L, MP_EYE_BLINK_R],
        "blink_lo": torch.tensor(lo_list, dtype=torch.float32),
        "blink_hi": torch.tensor(hi_list, dtype=torch.float32),
    }


def _calibrate_eye_channel(raw_v: float, lo: float, hi: float) -> float:
    return float(np.clip((raw_v - lo) / (hi - lo + 1e-6), 0.0, 1.0))


def apply_mp_blendshape_calibration(
    raw: torch.Tensor,
    bshapes_mode: torch.Tensor,
    eye_cal: dict,
) -> torch.Tensor:
    """
    Default: ``(raw - mode).clamp(0,1)``; **eyeBlink** only gets median→0.9 range remap.
    """
    mode = bshapes_mode.to(device=raw.device, dtype=raw.dtype)
    out = (raw - mode).clamp(0.0, 1.0)
    if eye_cal is None or "blink_idx" not in eye_cal:
        return out
    for j, lo, hi in zip(eye_cal["blink_idx"], eye_cal["blink_lo"], eye_cal["blink_hi"]):
        out[j] = _calibrate_eye_channel(float(raw[j]), float(lo), float(hi))
    return out


def apply_mp_blendshape_calibration_np(raw: np.ndarray, bshapes_mode: torch.Tensor, eye_cal: dict) -> np.ndarray:
    return apply_mp_blendshape_calibration(
        torch.tensor(raw, dtype=torch.float32),
        bshapes_mode,
        eye_cal,
    ).numpy()


def eye_au_stats_note(
    blendshapes_stack: np.ndarray,
    bshapes_mode: torch.Tensor,
    eye_cal: dict = None,
):
    for name, idx in [
        ("eyeBlink_L", MP_EYE_BLINK_L),
        ("eyeBlink_R", MP_EYE_BLINK_R),
        ("eyeWide_L", MP_EYE_WIDE_L),
        ("eyeWide_R", MP_EYE_WIDE_R),
    ]:
        col = blendshapes_stack[:, idx]
        line = (
            f"  {name}: raw median={np.median(col):.3f} mode={float(bshapes_mode[idx]):.3f} "
            f"p90={np.percentile(col, 90):.3f} min={col.min():.3f}"
        )
        if eye_cal is not None and "Blink" in name and idx in eye_cal.get("blink_idx", []):
            k = eye_cal["blink_idx"].index(idx)
            lo = float(eye_cal["blink_lo"][k])
            hi = float(eye_cal["blink_hi"][k])
            cal_med = _calibrate_eye_channel(float(np.median(col)), lo, hi)
            cal_p90 = _calibrate_eye_channel(float(np.percentile(col, 90)), lo, hi)
            cal_max = _calibrate_eye_channel(float(np.max(col)), lo, hi)
            line += f" | blink cal lo={lo:.3f} hi={hi:.3f} median→{cal_med:.3f} p90→{cal_p90:.3f} max→{cal_max:.3f}"
        print(line)

