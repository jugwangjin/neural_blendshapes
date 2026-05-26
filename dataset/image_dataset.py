"""
Image-split dataset: ``root/{split}/{image,mask,semantic,...}`` (``Config.input_dir`` layout).

On init: run/cache MediaPipe + face_alignment (tqdm); **no-face frames are excluded** from
the index. ``bshapes_mode`` + ``eye_au_cal`` are baked from the **train** split into
``bshapes_mode.pt``; eval/test only load that file.
Sampling up-weights AU+pose outliers vs near-mean frames (not sequential / AU-index boost).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.dataset_util import (
    _load_img,
    _load_mask,
    apply_mp_blendshape_calibration,
    apply_mp_blendshape_calibration_np,
    compute_bshapes_mode,
    compute_distribution_weights,
    compute_eye_au_calibration,
    eye_au_stats_note,
    list_split_images,
    matrix_to_pose_feat,
    paths_for_image,
)
from utils.mediapipe_blendshapes import MP_EYE_BLINK_L, MP_EYE_BLINK_R, MP_EYE_WIDE_L, MP_EYE_WIDE_R
from dataset.frame_processor import build_split_cache
from dataset.mediapipe_cache import default_frame_dict, load_frame_npz


def _normalize_fa68(lm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """face_alignment pixels → normalized UV in [0, 1] (matches ``mp_landmarks_2d``)."""
    out = lm.clone()
    if out.numel() == 0 or width <= 0 or height <= 0:
        return out
    out[..., 0] = out[..., 0] / float(width)
    out[..., 1] = out[..., 1] / float(height)
    return out


def _resize_chw(img_hwc: torch.Tensor, image_size: int) -> torch.Tensor:
    import cv2

    if img_hwc.shape[0] == image_size and img_hwc.shape[1] == image_size:
        arr = img_hwc.numpy()
    else:
        arr = cv2.resize(img_hwc.numpy(), (image_size, image_size), interpolation=cv2.INTER_AREA)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)


def _resize_mask(mask_hwc: torch.Tensor, image_size: int) -> torch.Tensor:
    import cv2

    m = mask_hwc.numpy()
    if m.ndim == 3:
        m = m[..., 0]
    m = cv2.resize(m, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return torch.tensor(m, dtype=torch.float32)[None]


def _bshapes_mode_path(cfg, subject_root: Path) -> Path:
    p = subject_root / "bshapes_mode.pt"
    if p.is_file():
        return p
    return Path(cfg.mp_cache_dir) / subject_root.name / "bshapes_mode.pt"


def _semantic_to_train_tensors(semantic_path, image_size: int):
    """FLARE semantic png → part label, h-reg masks, legacy seg_label / skin_mask."""
    from dataset.flare_semantic import load_flare_semantic_tensors

    sem = load_flare_semantic_tensors(semantic_path, image_size)
    return sem


class ImageDataset(Dataset):
    def __init__(
        self,
        cfg,
        train=True,
        synthetic_if_empty=False,
        distribution_boost=True,
    ):
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.train = train
        self.distribution_boost = distribution_boost and train
        self.distribution_ratio = getattr(cfg, "distribution_sample_ratio", 0.35)
        self.synthetic_if_empty = synthetic_if_empty

        split = cfg.flare_train_split if train else cfg.flare_eval_split
        subject_root = Path(cfg.input_dir)
        all_images = list_split_images(subject_root, split)

        rebuild = getattr(cfg, "rebuild_mp_cache", False)
        self.image_paths, self.cache_paths = build_split_cache(
            cfg,
            split,
            all_images,
            rebuild=rebuild,
            face_landmarker_task=cfg.face_landmarker_task,
        )

        if len(self.image_paths) == 0:
            if synthetic_if_empty:
                self.image_paths = None
                self.cache_paths = None
                self.sample_weights = None
                self.bshapes_mode = torch.zeros(cfg.num_mp_blendshapes)
                self.eye_au_cal = None
                return
            raise RuntimeError(
                f"No valid faces in {subject_root / split}/image "
                f"(run with rebuild_mp_cache or check face_landmarker.task)"
            )

        cal_path = _bshapes_mode_path(cfg, subject_root)
        stack = None
        if cal_path.is_file():
            blob = torch.load(cal_path, weights_only=False)
            if isinstance(blob, dict) and "eye_au_cal" in blob:
                self.bshapes_mode = blob["bshapes_mode"]
                self.eye_au_cal = blob["eye_au_cal"]
            else:
                self.bshapes_mode = blob
                self.eye_au_cal = None
        else:
            self.bshapes_mode = None
            self.eye_au_cal = None

        if self.train and (self.bshapes_mode is None or self.eye_au_cal is None):
            stack = self._load_blendshape_stack()
            pct = getattr(cfg, "bshapes_mode_percentile", 2.5)
            if self.bshapes_mode is None:
                self.bshapes_mode = compute_bshapes_mode(stack, percentile=pct)
            if self.eye_au_cal is None:
                self.eye_au_cal = compute_eye_au_calibration(
                    stack,
                    self.bshapes_mode,
                    blink_lo_percentile=getattr(cfg, "eye_blink_lo_percentile", 10.0),
                    blink_hi_percentile=getattr(cfg, "eye_blink_hi_percentile", 98.0),
                    min_range=getattr(cfg, "eye_au_min_range", 0.4),
                )
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"bshapes_mode": self.bshapes_mode, "eye_au_cal": self.eye_au_cal},
                cal_path,
            )
            print(
                f"saved bshapes_mode ({pct}th pct) -> {cal_path} | "
                f"eyeBlink L/R={float(self.bshapes_mode[MP_EYE_BLINK_L]):.3f}/"
                f"{float(self.bshapes_mode[MP_EYE_BLINK_R]):.3f} "
                f"eyeWide L/R={float(self.bshapes_mode[MP_EYE_WIDE_L]):.3f}/"
                f"{float(self.bshapes_mode[MP_EYE_WIDE_R]):.3f}"
            )
            eye_au_stats_note(stack, self.bshapes_mode, self.eye_au_cal)
        elif not self.train:
            if self.bshapes_mode is None or self.eye_au_cal is None:
                raise FileNotFoundError(
                    f"{cal_path} missing or legacy tensor-only — run ImageDataset(train=True) first"
                )

        if stack is None:
            stack = self._load_blendshape_stack()
        self._build_distribution_weights(stack)

        self._ram_cache = {}

    def _load_blendshape_stack(self):
        rows = []
        for cp in self.cache_paths:
            rows.append(load_frame_npz(cp)["mp_blendshape"].numpy())
        return np.stack(rows, axis=0)

    def _build_distribution_weights(self, stack):
        pose_rows = []
        for cp in self.cache_paths:
            d = load_frame_npz(cp)
            pf = d.get("pose_feat")
            if pf is None:
                pf = matrix_to_pose_feat(d["mp_transform_matrix"])
            pose_rows.append(pf.numpy() if isinstance(pf, torch.Tensor) else pf)

        cal_stack = np.stack(
            [
                apply_mp_blendshape_calibration_np(stack[i], self.bshapes_mode, self.eye_au_cal)
                for i in range(stack.shape[0])
            ],
            axis=0,
        )
        pose_stack = np.stack(pose_rows, axis=0)
        self.sample_weights = compute_distribution_weights(
            cal_stack,
            pose_stack,
            coeffs_are_calibrated=True,
            var_eps=getattr(self.cfg, "distribution_var_eps", 5e-2),
            low_weight=getattr(self.cfg, "distribution_low_weight", 0.05),
            high_cap=getattr(self.cfg, "distribution_high_cap", 1.0),
        )

    def _sample_index(self, idx):
        n = len(self.image_paths)
        if not self.distribution_boost or self.sample_weights is None:
            return idx % n
        if np.random.rand() < self.distribution_ratio:
            p = self.sample_weights / self.sample_weights.sum()
            return int(np.random.choice(n, p=p))
        return idx % n

    def __len__(self):
        if self.image_paths is None:
            return max(self.cfg.iterations, 1)
        return len(self.image_paths)

    def _load_frame_assets(self, img_path: Path):
        paths = paths_for_image(img_path)
        img = _load_img(paths["image"])
        if img.shape[-1] == 4:
            mask = img[..., 3:4]
            img = img[..., :3]
        elif paths["mask"].is_file():
            mask = _load_mask(paths["mask"])
        else:
            mask = torch.ones_like(img[..., :1])
        mask = mask.clamp(0, 1)
        img = img * mask

        semantic_path = paths["semantic"] if paths["semantic"].is_file() else None
        return img, mask, semantic_path

    def __getitem__(self, idx):
        if self.image_paths is None:
            return default_frame_dict(torch.device("cpu"), self.image_size)

        j = self._sample_index(idx)
        cached_out = self._get_frame_dict(j)
        
        # Shallow copy to update dynamically requested frame_idx safely
        out = dict(cached_out)
        out["frame_idx"] = idx
        return out

    def _get_frame_dict(self, j):
        if j in self._ram_cache:
            return self._ram_cache[j]

        img_path = self.image_paths[j]
        cache_path = self.cache_paths[j]
        mp = load_frame_npz(cache_path)

        img, mask, semantic_path = self._load_frame_assets(img_path)
        h0, w0 = int(img.shape[0]), int(img.shape[1])
        image = _resize_chw(img, self.image_size)
        mask = _resize_mask(mask, self.image_size)

        bs = apply_mp_blendshape_calibration(
            mp["mp_blendshape"],
            self.bshapes_mode,
            getattr(self, "eye_au_cal", None),
        )

        mp_T = mp["mp_transform_matrix"]
        mp_pose_raw = matrix_to_pose_feat(mp_T)[:6].float()

        out = {
            "image": image,
            "mask": mask,
            "mp_blendshape": bs,
            "mp_blendshape_raw": mp["mp_blendshape"],
            "mp_landmarks_2d": mp["mp_landmarks_2d"],
            "mp_landmarks_3d": mp["mp_landmarks_3d"],
            "mp_pose_raw": mp_pose_raw,
            "mp_transform_matrix": mp_T,
            "mp_valid": mp["mp_valid"],
            "landmark": _normalize_fa68(mp.get("landmark_fa", torch.zeros(68, 4)), w0, h0),
            "path": str(cache_path),
            "img_path": str(img_path),
            "frame_name": img_path.stem,
        }

        if semantic_path is not None:
            sem = _semantic_to_train_tensors(semantic_path, self.image_size)
            out["part_label"] = sem["part_label"]
            out["seg_label"] = sem["seg_label"]
            out["skin_mask"] = sem["skin_mask"]
            out["h_reg_skin"] = sem["h_reg_skin"]
            out["h_reg_eye"] = sem["h_reg_eye"]
            out["h_reg_brow"] = sem["h_reg_brow"]
            out["h_reg_misc"] = sem["h_reg_misc"]
            out["h_reg_mouth"] = sem["h_reg_mouth"]
            out["semantic_fg"] = sem["fg"]
            out["part_onehot"] = sem["part_onehot"]
            k = int(sem["seg_label"].max().item()) + 1
            out["seg_onehot"] = torch.nn.functional.one_hot(sem["seg_label"], num_classes=k).permute(2, 0, 1).float()

        self._ram_cache[j] = out
        return out