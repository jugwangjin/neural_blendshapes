"""
Disk cache for GT mask EDT distance fields (independent of MediaPipe ``.npz``).

Layout: ``{mask_edt_cache_dir}/{subject}/{scene}/{stem}.npz``
Keys: ``dist_out``, ``dist_in``, ``image_size``, ``downsample``, ``normalize``
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset.dataset_util import _load_img, _load_mask, paths_for_image, scene_tag_from_image
from dataset.mask_distance import compute_mask_distance_fields


def mask_edt_cache_path(cfg, img_path: Path) -> Path:
    subject = Path(cfg.input_dir).name
    scene = scene_tag_from_image(cfg.input_dir, img_path)
    root = Path(getattr(cfg, "mask_edt_cache_dir", Path("cache/mask_edt")))
    return root / subject / scene / f"{Path(img_path).stem}.npz"


def _cache_meta(cfg) -> dict:
    return {
        "image_size": int(cfg.image_size),
        "downsample": int(getattr(cfg, "mask_edt_downsample", 4)),
        "normalize": bool(getattr(cfg, "mask_edt_normalize", True)),
    }


def _meta_matches(d, meta: dict) -> bool:
    for k, v in meta.items():
        if k not in d:
            return False
        stored = d[k]
        if isinstance(v, bool):
            if bool(stored) != v:
                return False
        elif int(stored) != int(v):
            return False
    return True


def load_mask_distance_npz(path):
    d = np.load(path, allow_pickle=False)
    return {
        "mask_dist_out": torch.tensor(d["dist_out"], dtype=torch.float32),
        "mask_dist_in": torch.tensor(d["dist_in"], dtype=torch.float32),
    }


def save_mask_distance_npz(path, dist_out, dist_in, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dist_out=dist_out.detach().cpu().numpy().astype(np.float32),
        dist_in=dist_in.detach().cpu().numpy().astype(np.float32),
        image_size=np.int32(meta["image_size"]),
        downsample=np.int32(meta["downsample"]),
        normalize=np.bool_(meta["normalize"]),
    )


def load_mask_from_image_path(img_path: Path):
    paths = paths_for_image(img_path)
    img = _load_img(paths["image"])
    if img.shape[-1] == 4:
        mask = img[..., 3:4]
    elif paths["mask"].is_file():
        mask = _load_mask(paths["mask"])
    else:
        mask = torch.ones_like(img[..., :1])
    return mask.clamp(0, 1)


def load_or_compute_mask_distance(
    cfg, img_path: Path, mask_resized: torch.Tensor = None, *, rebuild: bool = False
):
    """
    Load cached EDT fields or compute from GT mask, save, return ``(dist_out, dist_in)`` [1,H,W].
    """
    meta = _cache_meta(cfg)
    out_npz = mask_edt_cache_path(cfg, img_path)
    force = rebuild or getattr(cfg, "rebuild_mask_edt_cache", False)

    if out_npz.is_file() and not force:
        d = np.load(out_npz, allow_pickle=False)
        if _meta_matches(d, meta):
            return (
                torch.tensor(d["dist_out"], dtype=torch.float32),
                torch.tensor(d["dist_in"], dtype=torch.float32),
            )

    if mask_resized is None:
        import cv2

        mask = load_mask_from_image_path(img_path)
        m = mask.numpy()
        if m.ndim == 3:
            m = m[..., 0]
        sz = int(cfg.image_size)
        if m.shape[0] != sz or m.shape[1] != sz:
            m = cv2.resize(m, (sz, sz), interpolation=cv2.INTER_NEAREST)
        mask_resized = torch.tensor(m, dtype=torch.float32)[None]

    dist_out, dist_in = compute_mask_distance_fields(
        mask_resized,
        meta["image_size"],
        downsample=meta["downsample"],
        normalize=meta["normalize"],
    )
    save_mask_distance_npz(out_npz, dist_out, dist_in, meta)
    return dist_out, dist_in


def build_mask_edt_cache(cfg, image_paths, *, rebuild: bool = False):
    """Precompute EDT npz for every image path (separate tqdm from MP cache)."""
    for img_path in tqdm(image_paths, desc="mask EDT cache"):
        load_or_compute_mask_distance(cfg, Path(img_path), rebuild=rebuild)


def default_mask_distance_fields(image_size: int, device=None):
    """Synthetic / smoke-test placeholders (not from MP cache)."""
    z = torch.zeros(1, image_size, image_size)
    if device is not None:
        z = z.to(device)
    return {"mask_dist_out": z, "mask_dist_in": z.clone()}
