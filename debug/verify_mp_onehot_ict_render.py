"""
Verify MP 52-channel → ICT expression mapping by one-hot renders (front camera).

For each MediaPipe cache column ``j`` (name from pkl inverse map):
  1. ``mp_onehot[j] = weight`` (default 1.0)
  2. ``mp_to_ict_expression_weights(mp)`` — ``mp[:, mediapipe_to_ict]`` → ICT [B, 53]
  3. ``ICT.forward`` → deformed mesh
  4. gsplat RGB render (training / default front camera)

Output PNG names use the MediaPipe expression name; duplicate names get ``__2``, ``__3``, …

Run from repo root (GPU server):

  python debug/verify_mp_onehot_ict_render.py
  python debug/verify_mp_onehot_ict_render.py --weight 0.75 --out_dir debug/out/mp_onehot_ict
  python debug/verify_mp_onehot_ict_render.py --channels 0,10,26 --save-obj
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEBUG = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DEBUG))

from config import Config
from model.gaussian_avatar import GaussianAvatar
from model.ict_model import ICTFaceKitTorch
from rendering import GaussianRenderer
from sanity.export_open3d import save_mesh_point_cloud
from utils.mediapipe_blendshapes import (
    NUM_MP_BLENDSHAPE_CHANNELS,
    load_mediapipe_mapping,
    mp_to_ict_expression_weights,
)
from utils.training_camera import load_training_camera, training_camera_status

PEACH_RGB = (0.88, 0.74, 0.64)
OPACITY_LOGIT = 12.0


def unique_mp_output_stems(mp_names):
    """One filename stem per MP cache channel (disambiguate duplicate names)."""
    seen: dict[str, int] = {}
    stems = []
    for name in mp_names:
        n = seen.get(name, 0)
        seen[name] = n + 1
        stems.append(name if n == 0 else f"{name}__{n + 1}")
    return stems


def safe_filename(stem: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", stem)


def save_rgb(path: Path, tensor_chw: torch.Tensor) -> np.ndarray:
    img = tensor_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    img = (img.clip(0, 1) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)
    return img


def rgb_to_logit(rgb):
    t = torch.tensor(rgb, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(t / (1.0 - t))


@torch.no_grad()
def setup_avatar(ict, device, *, k_face: int):
    avatar = GaussianAvatar.from_ict(
        ict,
        k_face=k_face,
        k_eyeball_sclera=2,
        k_eye_occlusion=2,
    ).to(device)
    avatar.color.data.copy_(rgb_to_logit(PEACH_RGB).expand_as(avatar.color.data))
    avatar.opacity.data.fill_(OPACITY_LOGIT)
    return avatar


@torch.no_grad()
def render_front(avatar, renderer, camera, verts, faces, device):
    out = avatar(verts=verts, faces=faces)
    render = renderer.render_rgb(out, camera, background=torch.zeros(3, device=device))
    return render["rgb"][0], out


def jaw_open_reference_verts(ict, device):
    """Camera framing mesh (jaw open, no other AUs)."""
    exp = torch.zeros(1, ict.num_expression, device=device)
    exp[0, ict.jaw_index] = float(ict.expression[0, ict.jaw_index].item())
    return ict.forward(
        expression_weights=exp,
        apply_flame_similarity=True,
        apply_eyeball_rotation=False,
    )[0]


def parse_channel_list(text: str | None) -> list[int]:
    if text is None or not str(text).strip():
        return list(range(NUM_MP_BLENDSHAPE_CHANNELS))
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        j = int(tok)
        if j < 0 or j >= NUM_MP_BLENDSHAPE_CHANNELS:
            raise ValueError(f"channel {j} out of range [0, {NUM_MP_BLENDSHAPE_CHANNELS})")
        out.append(j)
    return out


def ict_expr_label(ict, idx: int) -> str:
    names = ict.expression_names.tolist()
    if 0 <= idx < len(names):
        return str(names[idx])
    return f"expr_{idx}"


def main():
    parser = argparse.ArgumentParser(description="MP one-hot → ICT mesh front renders")
    parser.add_argument("--out_dir", type=str, default=str(DEBUG / "out" / "mp_onehot_ict"))
    parser.add_argument("--weight", type=float, default=1.0, help="One-hot MP coefficient")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--k_face", type=int, default=4, help="Gaussians per face tri (lower=faster)")
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma MP channel indices to render (default: all 52)",
    )
    parser.add_argument("--save-obj", action="store_true", help="Also write PLY point cloud per expression")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    cfg = Config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ict = ICTFaceKitTorch(
        npy_dir=str(cfg.ict_npy),
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
    ).to(device)

    mp_map = load_mediapipe_mapping(
        cfg.mediapipe_name_to_ict, num_expression=ict.num_expression
    )
    if not torch.equal(ict.mediapipe_to_ict.cpu(), mp_map.mediapipe_to_ict.cpu()):
        raise RuntimeError("ICT.mediapipe_to_ict buffer != rebuild from pkl")

    mp_names = mp_map.cache_channel_names
    mp_to_ict = mp_map.mediapipe_to_ict
    stems = unique_mp_output_stems(mp_names)
    channels = parse_channel_list(args.channels)

    renderer = GaussianRenderer(cfg, image_size=args.image_size, sh_degree=None).to(device)
    avatar = setup_avatar(ict, device, k_face=args.k_face)
    ref_verts = jaw_open_reference_verts(ict, device)
    camera = load_training_camera(
        ref_verts,
        path=cfg.camera_npz,
        width=args.image_size,
        height=args.image_size,
        device=device,
    )

    print(f"device={device}")
    print(f"out_dir={out_dir.resolve()}")
    print(f"camera: {training_camera_status(cfg.camera_npz)}")
    print(f"num_expression={ict.num_expression}  mp_channels={NUM_MP_BLENDSHAPE_CHANNELS}")
    print(f"rendering {len(channels)} channel(s), weight={args.weight}")

    rows = []
    faces = ict.faces.to(device)

    mp_zero = torch.zeros(1, NUM_MP_BLENDSHAPE_CHANNELS, device=device)
    exp_neutral = mp_to_ict_expression_weights(mp_zero, ict.mediapipe_to_ict, ict.num_expression)
    verts_neutral = ict.forward(
        expression_weights=exp_neutral,
        apply_flame_similarity=True,
        apply_eyeball_rotation=False,
    )[0]
    rgb_n, _ = render_front(avatar, renderer, camera, verts_neutral, faces, device)
    fn = "_neutral_mp_all_zero.png"
    save_rgb(out_dir / fn, rgb_n)
    rows.append(
        {
            "file": fn,
            "mp_channel": -1,
            "mp_name": "_neutral",
            "mp_weight": 0.0,
            "ict_index": 0,
            "ict_expr_name": ict_expr_label(ict, 0),
            "ict_weight": float(exp_neutral[0, 0].item()),
        }
    )

    weight = float(args.weight)
    for j in channels:
        mp_name = mp_names[j]
        stem = stems[j]
        fn = f"{safe_filename(stem)}.png"

        mp = torch.zeros(1, NUM_MP_BLENDSHAPE_CHANNELS, device=device)
        mp[0, j] = weight
        exp_w = mp_to_ict_expression_weights(mp, ict.mediapipe_to_ict, ict.num_expression)
        gather_slots = (mp_to_ict == j).nonzero(as_tuple=False).flatten()
        ict_idx = int(gather_slots[0].item()) if gather_slots.numel() else -1
        ict_w = float(exp_w[0, ict_idx].item()) if ict_idx >= 0 else 0.0

        verts = ict.forward(
            expression_weights=exp_w,
            apply_flame_similarity=True,
            apply_eyeball_rotation=False,
        )[0]
        rgb, avatar_out = render_front(avatar, renderer, camera, verts, faces, device)
        save_rgb(out_dir / fn, rgb)

        if args.save_obj:
            ply = out_dir / f"{safe_filename(stem)}.ply"
            save_mesh_point_cloud(ply, verts, PEACH_RGB, max_points=120000)

        rows.append(
            {
                "file": fn,
                "mp_channel": j,
                "mp_name": mp_name,
                "mp_weight": weight,
                "ict_index": ict_idx,
                "ict_expr_name": ict_expr_label(ict, ict_idx),
                "ict_weight": ict_w,
            }
        )
        print(
            f"  ch{j:02d} {mp_name:22s} → ICT[{ict_idx:2d}] {ict_expr_label(ict, ict_idx):22s}  "
            f"w={ict_w:.3f}  {fn}"
        )

    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    manifest_json = out_dir / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "mediapipe_pkl": str(cfg.mediapipe_name_to_ict),
                "ict_npy": str(cfg.ict_npy),
                "camera_npz": str(cfg.camera_npz),
                "weight": weight,
                "mp_to_ict": mp_to_ict.cpu().tolist(),
                "mp_cache_channel_names": mp_names,
                "output_stems": stems,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    grid_path = out_dir / "contact_sheet.png"
    _write_contact_sheet(out_dir, rows, grid_path, thumb=128)

    print(f"manifest: {manifest_csv}")
    print(f"contact:  {grid_path}")


def _write_contact_sheet(out_dir: Path, rows: list[dict], path: Path, thumb: int = 128):
    import cv2

    tiles = []
    labels = []
    for row in rows:
        p = out_dir / row["file"]
        if not p.is_file():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (thumb, thumb), interpolation=cv2.INTER_AREA)
        label = row["mp_name"] if row["mp_channel"] >= 0 else "_neutral"
        cv2.putText(
            im,
            f"{row['mp_channel']:02d} {label[:14]}",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(im)
        labels.append(label)

    if not tiles:
        return
    ncol = 8
    nrow = (len(tiles) + ncol - 1) // ncol
    while len(tiles) < nrow * ncol:
        tiles.append(np.zeros((thumb, thumb, 3), dtype=np.uint8))
    rows_img = []
    for r in range(nrow):
        rows_img.append(np.concatenate(tiles[r * ncol : (r + 1) * ncol], axis=1))
    sheet = np.concatenate(rows_img, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


if __name__ == "__main__":
    main()
