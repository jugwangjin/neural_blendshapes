"""
Image-split dataset: ``root/{scene}/image`` (``Config.input_dir`` layout).

``train_split`` / ``eval_split`` may be a scene name (``"train"``) or a list
(``["MVI_1797", "MVI_1801"]``); images from all scenes are merged into one index.

On init: run/cache MediaPipe + face_alignment (tqdm); **no-face frames are excluded** from
the index. ``bshapes_mode`` + ``eye_au_cal`` are baked from the **train** scenes into
``mp_coeff_mode.pt``; eval/test only load that file.
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
    merge_distribution_with_rgb_ema,
    compute_eye_au_calibration,
    eye_au_stats_note,
    format_splits_label,
    list_split_images,
    matrix_to_pose_feat,
    normalize_split_names,
    load_gt_normal,
    paths_for_image,
)
from utils.mediapipe_blendshapes import MP_EYE_BLINK_L, MP_EYE_BLINK_R, MP_EYE_WIDE_L, MP_EYE_WIDE_R
from dataset.frame_processor import build_split_cache
from dataset.mask_distance_cache import (
    build_mask_edt_cache,
    default_mask_distance_fields,
    load_or_compute_mask_distance,
)
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
    p = subject_root / "mp_coeff_mode.pt"
    if p.is_file():
        return p
    return Path(cfg.mp_cache_dir) / subject_root.name / "mp_coeff_mode.pt"


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
        self.seed = int(getattr(cfg, "seed", 0)) + (0 if train else 1)
        self.distribution_boost = distribution_boost and train
        self.distribution_ratio = getattr(cfg, "distribution_sample_ratio", 0.35)
        self.synthetic_if_empty = synthetic_if_empty

        split = cfg.train_split if train else cfg.eval_split
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

        if len(self.image_paths) > 0 and getattr(cfg, "precompute_mask_edt_cache", False):
            build_mask_edt_cache(
                cfg,
                self.image_paths,
                rebuild=getattr(cfg, "rebuild_mask_edt_cache", False),
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
                f"No valid faces under {subject_root}/{{{format_splits_label(split)}}}/image "
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
            pct = getattr(cfg, "bshapes_mode_percentile", 1.0)
            if self.bshapes_mode is None:
                self.bshapes_mode = compute_bshapes_mode(stack, percentile=pct)
            if self.eye_au_cal is None:
                self.eye_au_cal = compute_eye_au_calibration(
                    stack,
                    self.bshapes_mode,
                    blink_lo_percentile=getattr(cfg, "eye_blink_lo_percentile", 10.0),
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
        n = len(self.image_paths)
        self.rgb_loss_ema = np.zeros(n, dtype=np.float32)
        self._rgb_ema_path = _bshapes_mode_path(cfg, subject_root).parent / "rgb_loss_ema.pt"
        if getattr(cfg, "distribution_rgb_ema_enabled", False) and self._rgb_ema_path.is_file():
            blob = torch.load(self._rgb_ema_path, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and "ema" in blob:
                ema = np.asarray(blob["ema"], dtype=np.float32)
                if ema.shape[0] == n:
                    self.rgb_loss_ema = ema
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
        self.pca_sample_weights = compute_distribution_weights(
            cal_stack,
            pose_stack,
            coeffs_are_calibrated=True,
            var_eps=getattr(self.cfg, "distribution_var_eps", 5e-2),
            low_weight=getattr(self.cfg, "distribution_low_weight", 0.05),
            high_cap=getattr(self.cfg, "distribution_high_cap", 1.0),
        )
        self._rebuild_sample_weights()

    def _rebuild_sample_weights(self):
        low = float(getattr(self.cfg, "distribution_low_weight", 0.05))
        high = float(getattr(self.cfg, "distribution_high_cap", 1.0))
        if getattr(self.cfg, "distribution_rgb_ema_enabled", False):
            scale = float(getattr(self.cfg, "distribution_rgb_ema_scale", 1.0))
            max_lift = float(getattr(self.cfg, "distribution_rgb_ema_max_lift", 0.5))
            self.sample_weights = merge_distribution_with_rgb_ema(
                self.pca_sample_weights,
                self.rgb_loss_ema,
                ema_scale=scale,
                ema_max_lift=max_lift,
                low_weight=low,
                high_cap=high,
            )
        else:
            self.sample_weights = self.pca_sample_weights.copy()

    def update_rgb_loss_ema(self, dataset_frame_idx: int, rgb_l1: float):
        if not getattr(self.cfg, "distribution_rgb_ema_enabled", False):
            return
        j = int(dataset_frame_idx)
        v = float(rgb_l1)
        beta = float(getattr(self.cfg, "distribution_rgb_ema_beta", 0.995))
        prev = float(self.rgb_loss_ema[j])
        self.rgb_loss_ema[j] = v if prev <= 0.0 else beta * prev + (1.0 - beta) * v
        self._rebuild_sample_weights()

    def save_rgb_loss_ema(self):
        if not getattr(self.cfg, "distribution_rgb_ema_enabled", False):
            return
        self._rgb_ema_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"ema": self.rgb_loss_ema, "image_paths": [str(p) for p in self.image_paths]},
            self._rgb_ema_path,
        )

    def _sample_index(self, idx):
        n = len(self.image_paths)
        if not self.distribution_boost or self.sample_weights is None:
            return idx % n
        rng = self._sample_rng(idx)
        if rng.random() < self.distribution_ratio:
            p = self.sample_weights / self.sample_weights.sum()
            return int(rng.choice(n, p=p))
        return idx % n

    def _sample_rng(self, idx):
        worker_id = 0
        info = torch.utils.data.get_worker_info()
        if info is not None:
            worker_id = info.id
        return np.random.default_rng(int(self.seed) + worker_id * 1_000_003 + int(idx))

    def __len__(self):
        if self.image_paths is None:
            return max(self.cfg.iterations, 1)
        return len(self.image_paths)

    def _load_frame_assets(self, img_path: Path):
        paths = paths_for_image(img_path)
        img = _load_img(paths["image"])
        semantic_path = paths["semantic"] if paths["semantic"].is_file() else None
        if img.shape[-1] == 4:
            mask = img[..., 3:4]
            img = img[..., :3]
        elif paths["mask"].is_file():
            mask = _load_mask(paths["mask"])
        else:
            mask = torch.ones_like(img[..., :1])
        mask = mask.clamp(0, 1)
        from dataset.insta_tight_mask import apply_insta_tight_matting_mask

        mask = apply_insta_tight_matting_mask(mask, paths, self.cfg)
        img = img * mask

        normal_path = paths["normal"] if paths["normal"].is_file() else None
        return img, mask, semantic_path, normal_path

    def __getitem__(self, idx):
        if self.image_paths is None:
            out = default_frame_dict(torch.device("cpu"), self.image_size)
            out.update(default_mask_distance_fields(self.image_size))
            return out

        j = self._sample_index(idx)
        cached_out = self._get_frame_dict(j)
        
        # Shallow copy to update dynamically requested frame_idx safely
        out = dict(cached_out)
        out["frame_idx"] = idx
        out["dataset_frame_idx"] = j
        return out

    def _get_frame_dict(self, j):
        if j in self._ram_cache:
            return self._ram_cache[j]

        img_path = self.image_paths[j]
        cache_path = self.cache_paths[j]
        mp = load_frame_npz(cache_path)

        img, mask, semantic_path, normal_path = self._load_frame_assets(img_path)
        h0, w0 = int(img.shape[0]), int(img.shape[1])
        image = _resize_chw(img, self.image_size)
        mask = _resize_mask(mask, self.image_size)

        if getattr(self.cfg, "silhouette_use_edt", False) or getattr(
            self.cfg, "precompute_mask_edt_cache", False
        ):
            mask_dist_out, mask_dist_in = load_or_compute_mask_distance(
                self.cfg, img_path, mask_resized=mask
            )
        else:
            mask_dist_out, mask_dist_in = None, None

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
            "mask_dist_out": mask_dist_out,
            "mask_dist_in": mask_dist_in,
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
        if mask_dist_out is not None:
            out["mask_dist_out"] = mask_dist_out
            out["mask_dist_in"] = mask_dist_in

        out["gt_normal_valid"] = False
        if getattr(self.cfg, "load_gt_normals", True) and normal_path is not None:
            gt_normal = load_gt_normal(normal_path, self.image_size, mask=mask)
            if gt_normal is not None:
                out["gt_normal"] = gt_normal
                out["gt_normal_valid"] = True

        if getattr(self.cfg, "load_dataset_semantic", False) and semantic_path is not None:
            sem = _semantic_to_train_tensors(semantic_path, self.image_size)
            if getattr(self.cfg, "tight_mask_from_semantic", False):
                from dataset.flare_semantic import mask_gt_semantic_by_matting

                sem = mask_gt_semantic_by_matting(sem, mask)
            out["part_label"] = sem["part_label"]
            out["seg_label"] = sem["seg_label"]
            out["skin_mask"] = sem["skin_mask"]
            out["skin_tight"] = sem["skin_mask"]
            out["full_face_region_mask"] = sem["full_face_region_mask"]
            out["h_reg_label_eye_occlusion"] = sem["h_reg_label_eye_occlusion"]
            out["h_reg_seg_face"] = sem["h_reg_seg_face"]
            out["h_reg_seg_mouth"] = sem["h_reg_seg_mouth"]
            out["h_reg_seg_neck"] = sem["h_reg_seg_neck"]
            out["h_reg_seg_hair"] = sem["h_reg_seg_hair"]
            out["h_reg_seg_glasses"] = sem["h_reg_seg_glasses"]
            out["h_reg_seg_misc"] = sem["h_reg_seg_misc"]
            out["semantic_fg"] = sem["fg"]
            out["part_onehot"] = sem["part_onehot"]
            k = int(sem["seg_label"].max().item()) + 1
            out["seg_onehot"] = torch.nn.functional.one_hot(sem["seg_label"], num_classes=k).permute(2, 0, 1).float()

        self._ram_cache[j] = out
        return out
