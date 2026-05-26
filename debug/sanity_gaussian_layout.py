"""
Sanity check: fixed region colors + opacity≈1 → gsplat render (fixed camera).

Uses ``ICTFaceKitTorch.forward`` (single ``apply_flame_similarity``) — not raw ``neutral_mesh``.

Separate O(n) sweeps (not jaw×yaw Cartesian product):
  jaw sweep   — vary jawOpen, yaw=0
  yaw sweep   — vary head yaw, default jaw
  gaze sweep  — ICT eyeLook* at 0.75 activation, yaw=0

Surface Gaussians on MP-embedding triangles (``ict_lmk_face_idx``) are green;
other regions use flat/gradient debug colors (see ``sanity/region_colors.py``).

Run from repo root:
  python debug/sanity_gaussian_layout.py
  python debug/sanity_gaussian_layout.py --single
  python debug/sanity_gaussian_layout.py --sweep-jaw 0,0.5,1.0 --no-sweep-yaw --no-sweep-gaze
  python debug/sanity_gaussian_layout.py --sweep-gaze default --gaze-weight 0.75
"""

import argparse
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
from processing.ict_mediapipe_lmk.constants import DEFAULT_OUTPUT_LEGACY, DEFAULT_OUTPUT_NPZ
from processing.ict_mediapipe_lmk.embedding_io import resolve_embedding_path
from model.expr_regions import build_expr_region_weight
from model.gaussian_avatar import GaussianAvatar
from model.ict_deformer import ICTDeformer
from model.ict_model import ICTFaceKitTorch
from rendering import GaussianRenderer
from sanity.depth_vis import depth_alpha_from_render, depth_vis_images, overlay_rgb_depth
from sanity.export_open3d import save_gaussian_point_cloud, save_mesh_point_cloud
from sanity.region_colors import (
    mesh_vertex_rgb,
    rgb_to_logit,
    surface_gaussian_iris_stats,
    surface_gaussian_rgb,
)
from utils.camera import default_azimuth_sweep
from utils.sampling import count_surface_gaussians
from utils.camera import load_training_camera, training_camera_status

REGION_NAMES = {
    0: "mouth_interior",
    1: "mouth_socket",
    2: "eye_socket",
    3: "head_neck",
    4: "face",
    5: "eyeball_sclera",
    6: "eye_occlusion",
}

OPACITY_LOGIT = 12.0

DEFAULT_GAZE_EXPRS = [
    "eyeLookUp_L",
    "eyeLookDown_L",
    "eyeLookIn_L",
    "eyeLookOut_L",
    "eyeLookUp_R",
    "eyeLookDown_R",
    "eyeLookIn_R",
    "eyeLookOut_R",
]


def parse_float_list(text: str | None) -> list[float]:
    if text is None or not str(text).strip():
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_gaze_expr_list(text: str | None, ict) -> list[tuple[str, str]]:
    if text is None or not str(text).strip():
        return []
    raw = str(text).strip()
    if raw.lower() == "default":
        names = ict.expression_names.tolist()
        out = []
        for expr in DEFAULT_GAZE_EXPRS:
            if expr not in names:
                raise ValueError(f"unknown gaze expression {expr!r}")
            out.append((expr, expr))
        return out
    names = ict.expression_names.tolist()
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in names:
            raise ValueError(f"unknown gaze expression {token!r}; try --list-exprs")
        out.append((token, token))
    return out


def apply_surface_region_colors(avatar, ict, verts, device, *, mp_embedding_path):
    from utils.ict_regions import classify_surface_triangles_batch

    codes = classify_surface_triangles_batch(avatar.face_idx, ict.faces, ict, device)
    colors = surface_gaussian_rgb(
        avatar, ict, verts, device, mp_embedding_path=mp_embedding_path
    )
    avatar.color.data.copy_(rgb_to_logit(colors))

    counts = {}
    for code in range(-1, 7):
        c = int((codes == code).sum().item())
        if c > 0:
            counts[REGION_NAMES.get(code, f"code_{code}")] = c
    stats = surface_gaussian_iris_stats(avatar, ict, verts, device, mp_embedding_path=mp_embedding_path)
    counts["_iris_black_on_occ"] = stats["n_iris_black"]
    counts["_iris_faces_on_occ"] = stats["iris_faces_on_occ"]
    return counts


def apply_opacity_one(avatar):
    avatar.opacity.data.fill_(OPACITY_LOGIT)


def save_rgb(path, tensor_chw):
    img = tensor_chw.detach().float().cpu().permute(1, 2, 0).numpy()
    img = (img.clip(0, 1) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)
    return img


def save_depth_bundle(out_dir, tag, depth_out, rgb_uint8=None):
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


def build_expression_weights(ict, device, *, jaw=None, expr_name=None, expr_weight=0.0):
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


def mesh_for_render_deformer(deformer, ict, expression_weights, *, yaw_deg=0.0, apply_flame_similarity=True):
    verts = mesh_for_render(ict, expression_weights, apply_flame_similarity=apply_flame_similarity)
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
        aligned = mesh_for_render(ict, build_expression_weights(ict, device), apply_flame_similarity=True)
        no_align = mesh_for_render(ict, build_expression_weights(ict, device), apply_flame_similarity=False)
    _, _, ext_raw = mesh_bbox(raw)
    _, _, ext_aligned = mesh_bbox(aligned)
    _, _, ext_no = mesh_bbox(no_align)
    print(f"  bbox extent raw neutral:     {ext_raw:.6f}")
    print(f"  bbox extent jaw-open no align: {ext_no:.6f}")
    print(f"  bbox extent jaw-open + align:  {ext_aligned:.6f}")
    if abs(ext_raw - ext_aligned) < 1e-6 and info.get("use_flame_alignment"):
        print("  WARNING: raw vs aligned extent identical — check npy flame_alignment_*")
    flame_T = info.get("flame_T", [0.0, 0.0, 0.0])
    flame_active = info.get("use_flame_rigid") or abs(info.get("flame_s", 1.0) - 1.0) > 1e-8 or max(abs(t) for t in flame_T) > 1e-8
    if abs(ext_no - ext_aligned) < 1e-6 and flame_active:
        print("  WARNING: alignment appears inactive (extents match)")


@torch.no_grad()
def render_mesh(avatar, renderer, camera, verts, faces, device):
    out = avatar(verts=verts, faces=faces)
    render = renderer.render_rgb(out, camera, background=torch.zeros(3, device=device))
    depth_out = renderer.render_depth(out, camera, render_mode="ED")
    return render, depth_out, out


def build_render_jobs(args, ict, device):
    jobs = []

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
        jobs.append((tag, exp, 0.0, "single"))
        return jobs

    if args.sweep_expr_weight and args.expr:
        for w in parse_float_list(args.sweep_expr_weight):
            exp = build_expression_weights(ict, device, jaw=args.jaw, expr_name=args.expr, expr_weight=w)
            jobs.append((f"{args.expr}_{w:.3f}", exp, 0.0, "expr"))
        return jobs

    default_jaw = args.jaw
    for j in parse_float_list(args.sweep_jaw) if args.sweep_jaw is not None else []:
        jobs.append((f"jaw_{j:.3f}", build_expression_weights(ict, device, jaw=j), 0.0, "jaw"))

    yaws = []
    if args.sweep_yaw is not None:
        yaws = parse_float_list(args.sweep_yaw)
    elif args.sweep_azimuth is not None:
        yaws = parse_float_list(args.sweep_azimuth)
    elif args.sweep_azimuth_default:
        yaws = [float(a) for a in default_azimuth_sweep(args.azimuth_step)]
    ref_exp = build_expression_weights(ict, device, jaw=default_jaw)
    for yaw in yaws:
        jobs.append((f"yaw_{yaw:+.0f}", ref_exp, yaw, "yaw"))

    if args.sweep_gaze is not None:
        for tag, expr_name in parse_gaze_expr_list(args.sweep_gaze, ict):
            exp = build_expression_weights(
                ict, device, jaw=default_jaw, expr_name=expr_name, expr_weight=args.gaze_weight
            )
            jobs.append((f"gaze_{tag}", exp, 0.0, "gaze"))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Gaussian layout / FACS / alignment sanity render")
    parser.add_argument("--out", type=Path, default=DEBUG / "out" / "sanity_gaussians")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--jaw", type=float, default=None)
    parser.add_argument("--expr", type=str, default=None)
    parser.add_argument("--expr-weight", type=float, default=1.0)
    parser.add_argument("--sweep-jaw", type=str, default="0,0.5,1.0")
    parser.add_argument("--sweep-yaw", type=str, default="-30,0,30")
    parser.add_argument("--sweep-gaze", type=str, default="default")
    parser.add_argument("--gaze-weight", type=float, default=0.75)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--sweep-expr-weight", type=str, default="")
    parser.add_argument("--no-flame-align", action="store_true")
    parser.add_argument("--compare-raw-neutral", action="store_true")
    parser.add_argument("--list-exprs", action="store_true")
    parser.add_argument("--sweep-azimuth", type=str, default=None)
    parser.add_argument("--sweep-azimuth-default", action="store_true")
    parser.add_argument("--azimuth-step", type=float, default=30.0)
    parser.add_argument("--no-sweep-jaw", action="store_true")
    parser.add_argument("--no-sweep-yaw", action="store_true")
    parser.add_argument("--no-sweep-gaze", action="store_true")
    parser.add_argument("--no-save-pcd", action="store_true")
    parser.add_argument("--pcd-mode", type=str, default="gaussians", choices=("gaussians", "mesh", "both"))
    parser.add_argument("--pcd-max-points", type=int, default=100000)
    args = parser.parse_args()
    save_pcd = not args.no_save_pcd

    if args.no_sweep_jaw:
        args.sweep_jaw = ""
    if args.no_sweep_yaw:
        args.sweep_yaw = ""
    if args.no_sweep_gaze:
        args.sweep_gaze = ""

    cfg = Config()
    mp_emb = resolve_embedding_path(cfg.mp_embedding)
    if mp_emb.resolve() != Path(cfg.mp_embedding).resolve() and Path(cfg.mp_embedding).is_file():
        print(f"WARNING: cfg.mp_embedding missing; using {mp_emb}")
    elif (
        Path(cfg.mp_embedding).resolve() == DEFAULT_OUTPUT_LEGACY.resolve()
        and DEFAULT_OUTPUT_NPZ.is_file()
        and DEFAULT_OUTPUT_NPZ.resolve() != DEFAULT_OUTPUT_LEGACY.resolve()
    ):
        print(
            f"WARNING: cfg points to legacy {cfg.mp_embedding}; "
            f"fresh bake is {DEFAULT_OUTPUT_NPZ} — update config or pass matching file"
        )
    cfg.mp_embedding = mp_emb
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
        ict, ict.faces,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
    )
    print(f"device={device}  image_size={image_size}")
    print(f"mp_embedding: {cfg.mp_embedding}")
    print(f"surface Gaussians={n_surface}")

    deformer = ICTDeformer(ict, region_weight=build_expr_region_weight(ict).to(device)).to(device)
    deformer.eval()

    avatar = GaussianAvatar.from_ict(
        ict, deformer=deformer,
        k_face=cfg.n_surface_gaussians_per_face,
        k_head=cfg.n_surface_gaussians_per_head,
        k_mouth_socket=cfg.n_surface_gaussians_per_mouth_socket,
        k_mouth_interior=cfg.n_surface_gaussians_mouth_interior,
        k_eye_socket=cfg.n_surface_gaussians_per_eye_socket,
        k_eyeball_sclera=cfg.n_surface_gaussians_per_eyeball_sclera,
        k_eye_occlusion=cfg.n_surface_gaussians_per_eye_occlusion,
        n_semantic_classes=cfg.n_semantic_classes,
        gum_h_sigma_scale=cfg.gum_h_sigma_scale,
    ).to(device)

    apply_align = not args.no_flame_align
    ref_exp = build_expression_weights(ict, device, jaw=args.jaw)
    ref_verts = mesh_for_render_deformer(deformer, ict, ref_exp, yaw_deg=0.0, apply_flame_similarity=apply_align)
    region_counts = apply_surface_region_colors(avatar, ict, ref_verts, device, mp_embedding_path=cfg.mp_embedding)
    apply_opacity_one(avatar)

    print("surface Gaussians per region:")
    for name, c in sorted(region_counts.items()):
        if name.startswith("_"):
            print(f"  {name}: {c}")
        else:
            print(f"  {name}: {c}")

    renderer = GaussianRenderer(cfg, image_size=image_size, sh_degree=None).to(device)
    base_camera = load_training_camera(ref_verts, path=cfg.camera_npz, width=image_size, height=image_size, device=device)
    print(f"camera: {training_camera_status(cfg.camera_npz)}")

    render_jobs = build_render_jobs(args, ict, device)
    if not render_jobs:
        raise SystemExit("no render jobs — enable a sweep or use --single")

    n_jaw = sum(1 for *_, k in render_jobs if k == "jaw")
    n_yaw = sum(1 for *_, k in render_jobs if k == "yaw")
    n_gaze = sum(1 for *_, k in render_jobs if k == "gaze")
    print(f"render jobs: {len(render_jobs)} (jaw={n_jaw} yaw={n_yaw} gaze={n_gaze})")

    for tag, exp, yaw, kind in render_jobs:
        v_show = mesh_for_render_deformer(deformer, ict, exp, yaw_deg=yaw, apply_flame_similarity=apply_align)
        render, depth_out, avatar_out = render_mesh(avatar, renderer, base_camera, v_show, ict.faces, device)
        rgb_img = save_rgb(out_dir / f"{tag}_rgb.png", render["rgb"][0])
        save_depth_bundle(out_dir, tag, depth_out, rgb_uint8=rgb_img)
        extra = " depth+overlay"
        if save_pcd:
            seed = hash((tag, int(yaw))) % (2**31)
            if args.pcd_mode in ("gaussians", "both"):
                n_g = save_gaussian_point_cloud(out_dir / f"{tag}_gaussians.ply", avatar_out, max_points=args.pcd_max_points, seed=seed)
                extra += f" gauss_ply={n_g}"
            if args.pcd_mode in ("mesh", "both"):
                n_m = save_mesh_point_cloud(out_dir / f"{tag}_mesh_verts.ply", v_show, mesh_vertex_rgb(ict, v_show, device), max_points=args.pcd_max_points, seed=seed + 1)
                extra += f" mesh_ply={n_m}"
        print(f"wrote {tag}_rgb.png  kind={kind}  jawOpen={exp[0, ict.jaw_index].item():.4f}  yaw={yaw:+.0f}{extra}")

    if args.compare_raw_neutral:
        verts_raw = ict.neutral_mesh[0]
        render, depth_out, _ = render_mesh(avatar, renderer, base_camera, verts_raw, ict.faces, device)
        rgb_img = save_rgb(out_dir / "raw_neutral_rgb.png", render["rgb"][0])
        save_depth_bundle(out_dir, "raw_neutral", depth_out, rgb_uint8=rgb_img)

    legend = [
        "# Gaussian layout debug render",
        f"# camera: {training_camera_status(cfg.camera_npz)}",
        f"# surface={n_surface}",
        f"# mp_embedding: {cfg.mp_embedding}",
        "# green Gaussians = on faces with MP landmark bary embedding (ict_lmk_face_idx)",
    ]
    (out_dir / "README.txt").write_text("\n".join(legend) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
