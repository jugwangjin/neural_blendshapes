"""
Convert a ``train.py`` checkpoint (``.pt``) into a SuperSplat viewer PLY.

What we export:
  - **Neutral template face only**
  - **Ignore tracker + expression + pose**

How we build the template mesh:
  - verts = ict.template_reference_verts() + deformer.template_delta()

Output:
  - Standard 3DGS binary PLY (``x``, ``scale_*``, ``rot_*``, ``f_dc_*``, ``opacity``).
  - See https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/

Usage (from repo root):
  python scripts/pt_to_neutral_template_ply.py \
    --pt /path/to/checkpoint.pt \
    --out debug/justin.ply \
    --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from gsplat import export_splats

from rendering.pack import pack_gaussians
from model.build import avatar_checkpoint_layout_kwargs
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch


# exporter.sh2rgb(sh) = sh * C0 + 0.5
C0 = 0.28209479177387814


def _device_from_cli(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def main_export(
    *,
    pt_path: Path,
    out_path: Path,
    device: torch.device,
    max_points: int | None,
    format: str,
    seed: int,
):
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]

    # ---- build models (same construction as train.py) ----
    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)
    expr_region_weight = build_expr_region_weight(ict).to(device)

    deformer = ICTDeformer(
        ict,
        expr_region_weight,
        mediapipe_name_to_ict=str(cfg.mediapipe_name_to_ict),
        n_coeffs=cfg.num_ict_expressions,
    ).to(device)

    if "deformer" in payload and payload["deformer"] is not None:
        deformer.load_state_dict(payload["deformer"], strict=True)

    avatar_sd = payload.get("avatar")
    if avatar_sd is None:
        raise KeyError("checkpoint has no 'avatar' state_dict")

    avatar = GaussianAvatar.from_checkpoint_state(
        ict,
        deformer,
        avatar_sd,
        **avatar_checkpoint_layout_kwargs(cfg),
    ).to(device)

    # ---- neutral template verts (no pose / expression) ----
    template_delta = deformer.template_delta()  # [V,3]
    verts_template = ict.template_reference_verts() + template_delta  # [V,3]

    avatar_out = avatar(verts=verts_template, faces=ict.faces)
    # PLY f_dc: use SH DC band only (viewer has no view-dependent SH here).
    packed = pack_gaussians(avatar_out, sh_degree=None)

    means = packed["means"]
    scales = packed["scales"].clamp(min=1e-8)
    quats = packed["quats"]
    colors_rgb = packed["colors"]

    # 3DGS PLY stores log-scale and opacity logits (pre-sigmoid).
    scales_log = torch.log(scales)
    opacities_logits = avatar.opacity.reshape(-1)

    n = means.shape[0]
    if max_points is not None and max_points > 0 and max_points < n:
        g = torch.Generator(device=means.device)
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g, device=means.device)[: int(max_points)]
        means = means[idx]
        scales_log = scales_log[idx]
        quats = quats[idx]
        colors_rgb = colors_rgb[idx]
        opacities_logits = opacities_logits[idx]

    # ---- encode RGB -> SH degree-0 (f_dc_*) ----
    sh0 = ((colors_rgb - 0.5) / C0).to(dtype=torch.float32).reshape(-1, 1, 3)
    # No view-dependent SH in this avatar; gsplat still writes f_rest_* when K>0.
    shN = torch.zeros((sh0.shape[0], 0, 3), dtype=torch.float32, device=means.device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = export_splats(
        means=means.to(dtype=torch.float32),
        scales=scales_log.to(dtype=torch.float32),
        quats=quats.to(dtype=torch.float32),
        opacities=opacities_logits.to(dtype=torch.float32),
        sh0=sh0,
        shN=shN,
        format=format,
        save_to=None,
    )
    out_path.write_bytes(data)
    step = payload.get("step", "?")
    stage = payload.get("stage", "?")
    print(
        f"wrote {out_path}  (N={means.shape[0]} splats, step={step}, stage={stage}, format={format})"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=Path, required=True, help="train.py checkpoint (.pt)")
    p.add_argument("--out", type=Path, required=True, help="output ply for SuperSplat")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0 ...")
    p.add_argument("--max-points", type=int, default=120000, help="cap splats count (size/speed)")
    p.add_argument(
        "--format",
        type=str,
        default="ply",
        choices=("ply", "splat", "ply_compressed"),
        help="ply = standard 3DGS PLY for SuperSplat; ply_compressed uses packed_* (not standard PLY)",
    )
    p.add_argument("--seed", type=int, default=0, help="subset sampling seed")
    return p.parse_args()


def main():
    args = parse_args()
    device = _device_from_cli(args.device)
    main_export(
        pt_path=args.pt,
        out_path=args.out,
        device=device,
        max_points=args.max_points,
        format=args.format,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

