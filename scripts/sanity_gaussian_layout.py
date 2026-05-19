"""
Sanity check: fixed region colors + opacity≈1 → gsplat render (fixed camera).

Uses ``ICTFaceKitTorch.forward`` (single ``apply_flame_similarity``) — not raw ``neutral_mesh``.

Run from repo root (defaults: jaw 0,0.5,1.0 × gaze sweep × yaw -30,0,30):
  python scripts/sanity_gaussian_layout.py
  python scripts/sanity_gaussian_layout.py --out debugs/sanity_gaussians --image-size 512
  python scripts/sanity_gaussian_layout.py --expr smile --expr-weight 1.0
  python scripts/sanity_gaussian_layout.py --sweep-yaw -30,0,30
  python scripts/sanity_gaussian_layout.py --sweep-azimuth-default
  python scripts/sanity_gaussian_layout.py --single
  python scripts/sanity_gaussian_layout.py --no-save-pcd
  python scripts/sanity_gaussian_layout.py --pcd-mode both --pcd-max-points 50000
  # outputs per tag: *_rgb.png  *_depth.png  *_depth_gray.png  *_overlay.png  *_depth.npy
  python scripts/sanity_gaussian_layout.py --compare-raw-neutral
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from rendering import GaussianRenderer
from utils.camera import FixedCamera, default_azimuth_sweep
from utils.depth_vis import depth_alpha_from_render, depth_vis_images, overlay_rgb_depth
from utils.export_open3d import save_gaussian_point_cloud, save_mesh_point_cloud
from utils.sanity_region_colors import (
    eye_gaussian_rgb_shared,
    mesh_vertex_rgb,
    rgb_to_logit,
    surface_gaussian_rgb,
)
from utils.sampling import count_surface_gaussians

REGION_NAMES = {
    0: "mouth_interior",
    1: "mouth_socket",
    2: "eye_socket",
    3: "head_neck",
    4: "face",
}

ACCESSORY_RGB = (0.0, 1.0, 1.0)
OPACITY_LOGIT = 12.0


def apply_surface_region_colors(avatar, ict, verts, device):
    from utils.ict_regions import classify_surface_triangles_batch

    codes = classify_surface_triangles_batch(avatar.face_idx, ict.faces, ict, device)
    colors = surface_gaussian_rgb(avatar, ict, verts, device)
    avatar.color.data.copy_(rgb_to_logit(colors))

    counts = {}
    for code in range(-1, 5):
        c = int((codes == code).sum().item())
        if c > 0:
            counts[REGION_NAMES.get(code, f"code_{code}")] = c
    return counts


def apply_eye_colors(eyes, device):
    """Iris UV rect (black), sclera Gaussians (white); see ``eye_gaussian_rgb``."""
    colors = eye_gaussian_rgb_shared(eyes, device)
    eyes.color.data.copy_(rgb_to_logit(colors))


def apply_opacity_one(avatar):
    avatar.opacity.data.fill_(OPACITY_LOGIT)
    if avatar.eyes is not None:
        avatar.eyes.opacity.data.fill_(OPACITY_LOGIT)
    if avatar.accessory is not None and avatar.accessory.n_gaussians > 0:
        avatar.accessory.opacity.data.fill_(OPACITY_LOGIT)
        logit = rgb_to_logit(ACCESSORY_RGB)
        avatar.accessory.color.data.copy_(logit.unsqueeze(0).expand_as(avatar.accessory.color.data))


def save_rgb(path, tensor_chw):
    img = tensor_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    img = (img.clip(0, 1) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)
    return img


def save_depth_bundle(out_dir, tag, depth_out, rgb_uint8=None):
    """depth_vis (turbo), depth_gray, optional overlay; raw float32 .npy."""
    depth, alpha = depth_alpha_from_render(depth_out)
    depth_color, depth_gray = depth_vis_images(depth, alpha)
    stem = out_dir / tag
    stem.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(stem.with_name(f"{tag}_depth.png")), depth_color)
    imageio.imwrite(str(stem.with_name(f"{tag}_depth_gray.png")), depth_gray)
    np.save(str(stem.with_name(f"{tag}_depth.npy")), depth.detach().float().cpu().numpy())
    if rgb_uint8 is not None:
        imageio.imwrite(
            str(stem.with_name(f"{tag}_overlay.png")),
            overlay_rgb_depth(rgb_uint8, depth_color),
        )


def mesh_bbox(verts):
    v = verts.detach().float().cpu()
    mn = v.min(dim=0).values
    mx = v.max(dim=0).values
    ext = (mx - mn).norm().item()
    return mn.tolist(), mx.tolist(), ext


def build_expression_weights(ict, device, jaw=None, expr_name=None, expr_weight=0.0):
    exp = torch.zeros(1, ict.num_expression, device=device)
    jaw_val = float(jaw) if jaw is not None else float(ict.expression[0, ict.jaw_index].item())
    exp[0, ict.jaw_index] = jaw_val
    if expr_name is not None:
        names = ict.expression_names.tolist()
        if expr_name not in names:
            raise ValueError(f"unknown expression {expr_name!r}; try --list-exprs")
        exp[0, names.index(expr_name)] = float(expr_weight)
    return exp


def mesh_for_render(ict, expression_weights, *, apply_flame_similarity=True, to_canonical=False):
    return ict.forward(
        expression_weights=expression_weights,
        to_canonical=to_canonical,
        apply_flame_similarity=apply_flame_similarity,
        apply_eyeball_rotation=False,
    )[0]


def mesh_for_render_deformer(
    deformer,
    ict,
    expression_weights,
    *,
    yaw_deg=0.0,
    apply_flame_similarity=True,
):
    """ICT FACS via ``ict.forward``; optional world-Y yaw via ``deformer.apply_head_yaw``."""
    verts = mesh_for_render(
        ict, expression_weights, apply_flame_similarity=apply_flame_similarity
    )
    if float(yaw_deg) != 0.0:
        verts = deformer.apply_head_yaw(verts.unsqueeze(0), float(yaw_deg)).squeeze(0)
    return verts


def print_alignment_report(ict, device):
    info = ict.alignment_info()
    print("=== ICT alignment (npy neutral_mesh is raw OBJ; transform at forward) ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    with torch.no_grad():
        raw = ict.neutral_mesh[0]
        aligned = mesh_for_render(
            ict,
            build_expression_weights(ict, device),
            apply_flame_similarity=True,
        )
        no_align = mesh_for_render(
            ict,
            build_expression_weights(ict, device),
            apply_flame_similarity=False,
        )
    _, _, ext_raw = mesh_bbox(raw)
    _, _, ext_aligned = mesh_bbox(aligned)
    _, _, ext_no = mesh_bbox(no_align)
    print(f"  bbox extent raw neutral:     {ext_raw:.6f}")
    print(f"  bbox extent jaw-open no align: {ext_no:.6f}")
    print(f"  bbox extent jaw-open + align:  {ext_aligned:.6f}")
    if abs(ext_raw - ext_aligned) < 1e-6 and info.get("use_flame_alignment"):
        print("  WARNING: raw vs aligned extent identical — check npy flame_alignment_*")
    if abs(ext_no - ext_aligned) < 1e-6 and info.get("use_flame_similarity"):
        print("  WARNING: alignment appears inactive (extents match)")


def default_gaze_offsets(sweep_str: str):
    """Small bilateral gaze UV offsets: neutral + ±U / ±V for each magnitude in sweep."""
    vals = [float(x.strip()) for x in sweep_str.split(",") if x.strip()]
    offsets = [(0.0, 0.0)]
    seen = {(0.0, 0.0)}
    for g in vals:
        for du, dv in ((g, 0.0), (-g, 0.0), (0.0, g), (0.0, -g)):
            if du == 0.0 and dv == 0.0:
                continue
            pair = (du, dv)
            if pair not in seen:
                seen.add(pair)
                offsets.append(pair)
    return offsets


@torch.no_grad()
def render_mesh(
    avatar,
    renderer,
    camera,
    verts,
    faces,
    device,
    gaze_uv_left=None,
    gaze_uv_right=None,
):
    out = avatar(
        verts=verts,
        faces=faces,
        gaze_uv_left=gaze_uv_left,
        gaze_uv_right=gaze_uv_right,
    )
    render = renderer.render_rgb(out, camera, background=torch.zeros(3, device=device))
    depth_out = renderer.render_depth(out, camera, render_mode="ED")
    return render, depth_out, out


def main():
    parser = argparse.ArgumentParser(description="Gaussian layout / FACS / alignment sanity render")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "sanity_gaussians")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--jaw", type=float, default=None, help="jawOpen weight (default: npy flame_similarity_ict_jaw_open)")
    parser.add_argument("--expr", type=str, default=None, help="ICT expression name to activate")
    parser.add_argument("--expr-weight", type=float, default=1.0)
    parser.add_argument(
        "--sweep-jaw",
        type=str,
        default="0,0.5,1.0",
        help="Comma-separated jawOpen values (default: 0,0.5,1.0)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="One frame only (use --jaw / --expr; azimuth 0 only)",
    )
    parser.add_argument(
        "--sweep-expr-weight",
        type=str,
        default="",
        help="With --expr: comma-separated weights, e.g. 0,0.5,1.0",
    )
    parser.add_argument("--no-flame-align", action="store_true", help="forward with apply_flame_similarity=False")
    parser.add_argument(
        "--compare-raw-neutral",
        action="store_true",
        help="Also render raw neutral_mesh (no FACS, no alignment) for contrast",
    )
    parser.add_argument("--list-exprs", action="store_true", help="Print expression names and exit")
    parser.add_argument(
        "--sweep-gaze",
        type=str,
        default="0,0.03,-0.03",
        help="Gaze UV magnitudes (chart); neutral + ±U/±V per value (default: 0,0.03,-0.03)",
    )
    parser.add_argument(
        "--no-sweep-gaze",
        action="store_true",
        help="Only neutral gaze (0,0) on both eyes",
    )
    parser.add_argument(
        "--sweep-yaw",
        type=str,
        default="-30,0,30",
        help="Head yaw deg (world +Y) via ICTDeformer; fixed camera (default: -30,0,30)",
    )
    parser.add_argument(
        "--sweep-azimuth",
        type=str,
        default=None,
        help="Deprecated alias for --sweep-yaw",
    )
    parser.add_argument(
        "--sweep-azimuth-default",
        action="store_true",
        help="Also render -90..90 every 30° (left/right orbit)",
    )
    parser.add_argument(
        "--azimuth-step",
        type=float,
        default=30.0,
        help="Step for --sweep-azimuth-default (default 30)",
    )
    parser.add_argument(
        "--no-view-correction",
        action="store_true",
        help="Skip default Y180+roll180 (face toward camera, upright)",
    )
    parser.add_argument(
        "--no-save-pcd",
        action="store_true",
        help="Skip Open3D colored .ply export (default: save PLY)",
    )
    parser.add_argument(
        "--pcd-mode",
        type=str,
        default="gaussians",
        choices=("gaussians", "mesh", "both"),
        help="Point cloud source",
    )
    parser.add_argument(
        "--pcd-max-points",
        type=int,
        default=100000,
        help="Random subsample cap per PLY (Gaussians ~200k)",
    )
    args = parser.parse_args()
    save_pcd = not args.no_save_pcd

    cfg = Config()
    image_size = args.image_size or cfg.image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy)).to(device)

    if args.list_exprs:
        for i, name in enumerate(ict.expression_names.tolist()):
            print(f"{i:3d} {name}")
        return

    print_alignment_report(ict, device)

    n_surface = count_surface_gaussians(
        ict,
        ict.faces,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
    )
    n_eye = 2 * cfg.n_eye_gaussians_per_side
    print(f"device={device}  image_size={image_size}")
    print(f"surface Gaussians={n_surface}  eye Gaussians={n_eye}  accessory={cfg.n_accessory_gaussians}")

    expr_region_weight = build_expr_region_weight(ict).to(device)
    deformer = ICTDeformer(ict, region_weight=expr_region_weight).to(device)
    deformer.eval()

    avatar = GaussianAvatar.from_ict(
        ict,
        deformer=deformer,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        n_eye_per_side=cfg.n_eye_gaussians_per_side,
        n_accessory_gaussians=cfg.n_accessory_gaussians,
        n_semantic_classes=cfg.n_semantic_classes,
        gum_h_sigma_scale=cfg.gum_h_sigma_scale,
    ).to(device)

    apply_align = not args.no_flame_align
    ref_exp = build_expression_weights(ict, device, jaw=args.jaw, expr_name=args.expr, expr_weight=0.0)
    ref_verts = mesh_for_render_deformer(
        deformer, ict, ref_exp, yaw_deg=0.0, apply_flame_similarity=apply_align
    )

    region_counts = apply_surface_region_colors(avatar, ict, ref_verts, device)
    apply_eye_colors(avatar.eyes, device)
    apply_opacity_one(avatar)

    print("surface Gaussians per region:")
    for name, c in sorted(region_counts.items()):
        print(f"  {name}: {c}")

    renderer = GaussianRenderer(cfg, image_size=image_size, sh_degree=None).to(device)
    if cfg.camera_npz.is_file():
        base_camera = FixedCamera.from_default_npz(
            cfg.camera_npz, width=image_size, height=image_size, device=device
        )
        print(f"camera: {cfg.camera_npz}")
    else:
        base_camera = FixedCamera.from_mesh_bounds(ref_verts, width=image_size, height=image_size)
        print(
            f"camera: mesh-bounds fit (no {cfg.camera_npz}) — "
            f"run: python scripts/bake_default_camera.py"
        )

    if args.single:
        yaws = [0.0]
    elif args.sweep_azimuth_default:
        yaws = [float(a) for a in default_azimuth_sweep(args.azimuth_step)]
    else:
        sweep = args.sweep_yaw if args.sweep_yaw is not None else args.sweep_azimuth
        if sweep is None:
            sweep = "-30,0,30"
        yaws = [float(x.strip()) for x in sweep.split(",") if x.strip()]
    pivot = deformer.canonical_xyz.mean(dim=0)
    if not args.no_view_correction:
        base_camera = base_camera.with_view_correction(pivot)
        print("camera view correction: yaw=180° (world +Y) + roll=180° (cam +Z)")
    print(f"head yaw sweep (deformer.apply_head_yaw): {yaws}  camera=fixed")

    if args.no_sweep_gaze:
        gaze_offsets = [(0.0, 0.0)]
    else:
        gaze_offsets = default_gaze_offsets(args.sweep_gaze)
    print(f"gaze UV sweep (both eyes, shared offset): {gaze_offsets}")

    render_jobs = []

    if args.single:
        exp = build_expression_weights(
            ict, device, jaw=args.jaw, expr_name=args.expr, expr_weight=args.expr_weight
        )
        tag = "sanity"
        if args.jaw is not None:
            tag += f"_jaw{args.jaw:.3f}"
        if args.expr:
            tag += f"_{args.expr}{args.expr_weight:.2f}"
        if args.no_flame_align:
            tag += "_no_align"
        render_jobs.append((tag, exp))
    elif args.sweep_expr_weight and args.expr:
        weights = [float(x.strip()) for x in args.sweep_expr_weight.split(",") if x.strip()]
        for w in weights:
            exp = build_expression_weights(
                ict, device, jaw=args.jaw, expr_name=args.expr, expr_weight=w
            )
            render_jobs.append((f"{args.expr}_{w:.3f}", exp))
    else:
        jaws = [float(x.strip()) for x in args.sweep_jaw.split(",") if x.strip()]
        for j in jaws:
            exp = build_expression_weights(ict, device, jaw=j, expr_name=args.expr, expr_weight=0.0)
            tag = f"jaw_{j:.3f}"
            if args.expr:
                tag += f"_{args.expr}_0"
            render_jobs.append((tag, exp))

    for tag, exp in render_jobs:
        for gu, gv in gaze_offsets:
            gaze_l = torch.tensor([gu, gv], device=device, dtype=torch.float32)
            gaze_r = gaze_l.clone()
            tag_gaze = tag if gu == 0.0 and gv == 0.0 else f"{tag}_gaze{gu:+.3f}_{gv:+.3f}"
            for yaw in yaws:
                v_show = mesh_for_render_deformer(
                    deformer,
                    ict,
                    exp,
                    yaw_deg=yaw,
                    apply_flame_similarity=apply_align,
                )
                tag_az = tag_gaze if yaw == 0.0 else f"{tag_gaze}_yaw{yaw:+.0f}"
                render, depth_out, avatar_out = render_mesh(
                    avatar,
                    renderer,
                    base_camera,
                    v_show,
                    ict.faces,
                    device,
                    gaze_uv_left=gaze_l,
                    gaze_uv_right=gaze_r,
                )
                rgb_img = save_rgb(out_dir / f"{tag_az}_rgb.png", render["rgb"][0])
                save_depth_bundle(out_dir, tag_az, depth_out, rgb_uint8=rgb_img)
                extra = " depth+overlay"
                if save_pcd:
                    seed = hash((tag_az, int(yaw), gu, gv)) % (2**31)
                    if args.pcd_mode in ("gaussians", "both"):
                        n_g = save_gaussian_point_cloud(
                            out_dir / f"{tag_az}_gaussians.ply",
                            avatar_out,
                            max_points=args.pcd_max_points,
                            seed=seed,
                        )
                        extra += f" gauss_ply={n_g}"
                    if args.pcd_mode in ("mesh", "both"):
                        mesh_rgb = mesh_vertex_rgb(ict, v_show, device)
                        n_m = save_mesh_point_cloud(
                            out_dir / f"{tag_az}_mesh_verts.ply",
                            v_show,
                            mesh_rgb,
                            max_points=args.pcd_max_points,
                            seed=seed + 1,
                        )
                        extra += f" mesh_ply={n_m}"
                print(
                    f"wrote {tag_az}_rgb.png  jawOpen={exp[0, ict.jaw_index].item():.4f}  "
                    f"gaze=({gu:+.3f},{gv:+.3f})  yaw={yaw:+.0f}  align={apply_align}{extra}"
                )

    if args.compare_raw_neutral:
        verts_raw = ict.neutral_mesh[0]
        render, depth_out, _ = render_mesh(avatar, renderer, base_camera, verts_raw, ict.faces, device)
        rgb_img = save_rgb(out_dir / "raw_neutral_rgb.png", render["rgb"][0])
        save_depth_bundle(out_dir, "raw_neutral", depth_out, rgb_uint8=rgb_img)
        print("wrote raw_neutral_rgb.png + depth (no jawOpen, no flame_alignment — expect wrong scale/pose)")

    legend = [
        "# Gaussian layout sanity (fixed camera; head yaw + bilateral gaze UV sweep)",
        "# mesh = ICTDeformer → ICT FACS + optional flame_alignment + head_yaw_deg",
        "# face=peach Y-gradient (high Y bright)  head/neck=white↔gray Y-gradient  eye=white",
        "# mouth_interior=red  mouth_socket=dark red  eye_socket=slate",
        "# per frame: *_rgb.png  *_depth.png (turbo, alpha-masked)  *_depth_gray.png  *_overlay.png  *_depth.npy",
        f"# surface={n_surface} eye={n_eye}",
        f"# alignment: {ict.alignment_info()}",
    ]
    for line in legend:
        print(line)
    (out_dir / "README.txt").write_text("\n".join(legend) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
