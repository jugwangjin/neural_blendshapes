"""Video dataset from precomputed MediaPipe + mask caches."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.mediapipe_cache import default_frame_dict, load_frame_npz, list_cached_frames


class VideoDataset(Dataset):
    def __init__(self, cfg, train=True, synthetic_if_empty=True, au_active_boost=False):
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.au_active_boost = au_active_boost and train
        self.au_active_ratio = getattr(cfg, "au_active_sample_ratio", 0.3)
        self.au_active_thresh = getattr(cfg, "au_active_thresh", 0.12)
        from dataset.dataset_util import normalize_split_names
        split = cfg.train_split if train else cfg.eval_split
        scenes = normalize_split_names(split)
        self.frames = []
        for scene in scenes:
            scene_dir = Path(cfg.input_dir) / scene / cfg.mp_cache_dir.name
            if scene_dir.exists():
                self.frames.extend(list_cached_frames(scene_dir))
            else:
                alt = Path(cfg.mp_cache_dir) / scene
                if alt.exists():
                    self.frames.extend(list_cached_frames(alt))

        self.synthetic_if_empty = synthetic_if_empty
        if len(self.frames) == 0 and synthetic_if_empty:
            self.frames = None

        self.high_au_indices = []
        if self.frames is not None and self.au_active_boost:
            self._build_high_au_indices()

    def _build_high_au_indices(self):
        thresh = self.au_active_thresh
        for i, path in enumerate(self.frames):
            mp = load_frame_npz(path)
            bs = mp.get("mp_blendshape")
            if bs is None:
                continue
            if float(bs.max()) > thresh:
                self.high_au_indices.append(i)
        if len(self.high_au_indices) == 0:
            self.high_au_indices = list(range(len(self.frames)))

    def _sample_index(self, idx):
        if (
            self.au_active_boost
            and len(self.high_au_indices) > 0
            and np.random.rand() < self.au_active_ratio
        ):
            return int(np.random.choice(self.high_au_indices))
        return idx % len(self.frames)

    def __len__(self):
        if self.frames is None:
            return max(self.cfg.iterations, 1)
        return len(self.frames)

    def _load_segmentation(self, npz_path):
        stem = Path(npz_path).with_suffix("")
        seg_path = stem.with_name(stem.name + "_seg.png")
        if not seg_path.exists():
            seg_cache = Path(self.cfg.segmentation_dir) / stem.name / (stem.name + "_seg.png")
            if seg_cache.exists():
                seg_path = seg_cache
        if not seg_path.exists():
            return None, None

        import cv2

        seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        seg = cv2.resize(seg, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        seg_label = torch.tensor(seg, dtype=torch.long)
        k = int(seg_label.max().item()) + 1
        seg_onehot = torch.nn.functional.one_hot(seg_label, num_classes=k).permute(2, 0, 1).float()
        return seg_label, seg_onehot

    def _load_image_mask(self, npz_path):
        stem = Path(npz_path).with_suffix("")
        img_path = stem.with_name(stem.name + "_rgb.png")
        mask_path = stem.with_name(stem.name + "_mask.png")
        if img_path.exists():
            import cv2

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            image = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            image = torch.rand(3, self.image_size, self.image_size)

        if mask_path.exists():
            import cv2

            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (self.image_size, self.image_size))
            mask = torch.tensor(m, dtype=torch.float32)[None] / 255.0
        else:
            mask = torch.ones(1, self.image_size, self.image_size)
        return image, mask

    def __getitem__(self, idx):
        if self.frames is None:
            return default_frame_dict(torch.device("cpu"), self.image_size)

        path = self.frames[self._sample_index(idx)]
        mp = load_frame_npz(path)
        image, mask = self._load_image_mask(path)
        seg_label, seg_onehot = self._load_segmentation(path)
        out = {
            "image": image,
            "mask": mask,
            **mp,
            "frame_idx": idx,
            "path": str(path),
        }
        if seg_label is not None:
            out["seg_label"] = seg_label
            out["seg_onehot"] = seg_onehot
        return out


def collate_batch(items):
    batch = {}
    for key in items[0]:
        if key in ("path", "frame_idx"):
            batch[key] = [x[key] for x in items]
            continue
        vals = [x[key] for x in items]
        if isinstance(vals[0], torch.Tensor):
            batch[key] = torch.stack(vals, dim=0)
        else:
            batch[key] = vals
    return batch
