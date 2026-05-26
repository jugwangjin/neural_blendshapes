"""
Compute camera translation so frontal ICT Multi-PIE-68 landmarks match metrical-tracker crop.

``ICTFaceKitTorch.canonical`` = jawOpen + npy ``flame_alignment_s,R,T`` (same as bake).
Keeps ``R_mean`` / ``K_mean`` from ``assets/default_camera.npz``; solves ``t_mean`` only.

Run from repo root:
  python processing/compute_camera_for_metrical_crop.py
  python processing/compute_camera_for_metrical_crop.py --apply-train-view --write-npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PROCESSING = Path(__file__).resolve().parent
_REPO = _PROCESSING.parent
for _p in (_REPO, _PROCESSING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import Config
from model.ict_model import ICTFaceKitTorch
from processing.metrical_crop_landmarks import (
    metrical_crop_landmarks,
    target_landmark_half_span,
)
from processing.paths import setup_import_paths
from utils.camera import FixedCamera
from utils.camera import load_default_camera, save_default_camera

setup_import_paths()


def _scale_K(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    sh, sw = src_hw
    dh, dw = dst_hw
    K = np.asarray(K, dtype=np.float64).copy()
    sx = dw / float(sw)
    sy = dh / float(sh)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K


def load_pie68_3d(ict_npy: Path) -> tuple[np.ndarray, dict]:
    """Multi-PIE 68 on ``ict.canonical`` (FLAME space from npy flame alignment)."""
    ict = ICTFaceKitTorch(npy_dir=str(ict_npy))
    mesh = ict.canonical[0]
    lmk = ict.landmark_vertices(mesh, region="all").detach().cpu().numpy()
    return lmk.astype(np.float64), ict.alignment_info()


def load_R_K(
    camera_npz: Path,
    *,
    image_size: int,
    apply_train_view: bool,
    pivot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, K, t_init) — R/K fixed; t_init for optimizer warm-start only."""
    data = load_default_camera(camera_npz)
    if data is None:
        raise FileNotFoundError(f"camera npz not found: {camera_npz}")
    R = np.asarray(data["R_mean"], dtype=np.float64)
    K = np.asarray(data["K_mean"], dtype=np.float64)
    t_init = np.asarray(data["t_mean"], dtype=np.float64).reshape(3)

    baked_hw = (int(K[1, 2] * 2), int(K[0, 2] * 2))
    dst_hw = (image_size, image_size)
    if baked_hw != dst_hw:
        K = _scale_K(K, baked_hw, dst_hw)

    if apply_train_view:
        cam = FixedCamera(
            width=image_size,
            height=image_size,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            R=torch.tensor(R, dtype=torch.float32),
            t=torch.tensor(t_init, dtype=torch.float32),
        )
        pivot_t = torch.tensor(pivot, dtype=torch.float32)
        cam = cam.with_view_correction(pivot_t, yaw_deg=180.0, roll_deg=180.0)
        R = cam.R.detach().cpu().numpy().astype(np.float64)
        t_init = cam.t.detach().cpu().numpy().astype(np.float64)

    return R, K, t_init


def project_landmarks(
    lmk3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV row convention: p_cam = p @ R.T + t; u = fx*x/z+cx."""
    pc = lmk3d @ R.T + t.reshape(1, 3)
    z = np.clip(pc[:, 2], 1e-4, None)
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), pc


def _landmark_metrics(uv_crop: np.ndarray, out_size: int, bb_scale: float) -> dict:
    center = uv_crop.mean(axis=0)
    target = np.array([out_size * 0.5, out_size * 0.5], dtype=np.float64)
    half = max(
        float(uv_crop[:, 0].max() - center[0]),
        float(center[0] - uv_crop[:, 0].min()),
        float(uv_crop[:, 1].max() - center[1]),
        float(center[1] - uv_crop[:, 1].min()),
    )
    tgt_half = target_landmark_half_span(out_size, bb_scale)
    return {
        "center": center.tolist(),
        "center_err": float(np.linalg.norm(center - target)),
        "half_span": half,
        "target_half_span": tgt_half,
        "span_err": float(half - tgt_half),
    }


def assess_crop_quality(metrics: dict, *, center_tol: float = 12.0, span_tol: float = 12.0) -> dict:
    center_err = float(metrics["center_err"])
    span_err = abs(float(metrics["span_err"]))
    ok = center_err <= center_tol and span_err <= span_tol
    return {
        "acceptable": ok,
        "center_tol": center_tol,
        "span_tol": span_tol,
        "center_err": center_err,
        "span_err": span_err,
    }


def solve_translation(
    lmk3d: np.ndarray,
    R: np.ndarray,
    K: np.ndarray,
    t_init: np.ndarray,
    *,
    virtual_hw: tuple[int, int],
    bb_scale: float,
    out_size: int,
    w_center: float = 1.0,
    w_span: float = 0.35,
    w_depth: float = 10.0,
    min_depth: float = 0.05,
) -> tuple[np.ndarray, dict]:
    from scipy.optimize import minimize

    t0 = np.asarray(t_init, dtype=np.float64).reshape(3)
    h_virt, w_virt = virtual_hw

    def objective(t_vec: np.ndarray) -> float:
        uv, pc = project_landmarks(lmk3d, R, t_vec, K)
        if pc[:, 2].min() < min_depth:
            return 1e6 + float(min_depth - pc[:, 2].min()) * w_depth
        uv_crop, _ = metrical_crop_landmarks(
            uv, (h_virt, w_virt), bb_scale=bb_scale, out_size=out_size
        )
        center = uv_crop.mean(axis=0)
        target = np.array([out_size * 0.5, out_size * 0.5])
        loss_c = float(np.sum((center - target) ** 2))
        half = max(
            float(uv_crop[:, 0].max() - center[0]),
            float(center[0] - uv_crop[:, 0].min()),
            float(uv_crop[:, 1].max() - center[1]),
            float(center[1] - uv_crop[:, 1].min()),
        )
        tgt_half = target_landmark_half_span(out_size, bb_scale)
        loss_s = (half - tgt_half) ** 2
        return w_center * loss_c + w_span * loss_s

    res = minimize(objective, t0, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9})
    t_opt = np.asarray(res.x, dtype=np.float64)
    if not res.success or res.nit <= 1:
        res2 = minimize(objective, t_opt, method="Powell", options={"maxiter": 400, "ftol": 1e-8})
        if float(res2.fun) < float(res.fun):
            res = res2
            t_opt = np.asarray(res.x, dtype=np.float64)
    uv, _ = project_landmarks(lmk3d, R, t_opt, K)
    uv_crop, bbox = metrical_crop_landmarks(
        uv, (h_virt, w_virt), bb_scale=bb_scale, out_size=out_size
    )
    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "fun": float(res.fun),
        "t_init": t0.tolist(),
        "t_opt": t_opt.tolist(),
        "virtual_hw": [h_virt, w_virt],
        "bbox": bbox.tolist(),
        "metrics": _landmark_metrics(uv_crop, out_size, bb_scale),
    }
    return t_opt, info


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Solve camera translation for metrical-tracker crop framing."
    )
    parser.add_argument("--ict-npy", type=Path, default=cfg.ict_npy)
    parser.add_argument("--camera-npz", type=Path, default=cfg.camera_npz)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--bbox-scale", type=float, default=2.5)
    parser.add_argument(
        "--virtual-size",
        type=int,
        default=8192,
        help="Dummy full-frame (H=W) for get_bbox before crop (landmarks only)",
    )
    parser.add_argument(
        "--apply-train-view",
        action="store_true",
        help="Compose yaw=180 roll=180 on R (same as train.py with_view_correction)",
    )
    parser.add_argument(
        "--write-npz",
        action="store_true",
        help="Write camera npz with new t_mean (R_mean/K_mean unchanged)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output npz path (default: overwrite --camera-npz if --write-npz)",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    lmk3d, align_info = load_pie68_3d(args.ict_npy)
    pivot = lmk3d.mean(axis=0)
    R, K, t_init = load_R_K(
        args.camera_npz,
        image_size=args.image_size,
        apply_train_view=args.apply_train_view,
        pivot=pivot,
    )

    virtual_hw = (args.virtual_size, args.virtual_size)
    t_opt, info = solve_translation(
        lmk3d,
        R,
        K,
        t_init,
        virtual_hw=virtual_hw,
        bb_scale=args.bbox_scale,
        out_size=args.image_size,
    )

    print("=== Camera for metrical crop ===")
    print(f"  ict_npy: {args.ict_npy}")
    print(
        f"  flame map (npy): rigid={align_info['use_flame_rigid']} "
        f"s={align_info['flame_s']:.6f} jawOpen={align_info['flame_similarity_ict_jaw_open']:.4f}"
    )
    print(f"  camera_npz (R,K source): {args.camera_npz}")
    print(f"  apply_train_view: {args.apply_train_view}")
    print(f"  bbox_scale: {args.bbox_scale}  image_size: {args.image_size}")
    print(f"  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(f"  t_init: {info['t_init']}")
    print(f"  t_opt:  {info['t_opt']}")
    m = info["metrics"]
    quality = assess_crop_quality(m)
    info["quality"] = quality
    print(
        f"  after crop: center={m['center']} center_err={m['center_err']:.3f} "
        f"half_span={m['half_span']:.2f} target={m['target_half_span']:.2f} "
        f"span_err={m['span_err']:.3f}"
    )
    print(f"  optimizer: success={info['success']} nit={info['nit']} fun={info['fun']:.6f}")
    print(
        f"  quality: acceptable={quality['acceptable']} "
        f"(center_err<={quality['center_tol']}, |span_err|<={quality['span_tol']})"
    )

    if args.write_npz:
        out_path = args.out or args.camera_npz
        save_default_camera(
            R,
            t_opt,
            K,
            out_path,
            train_view_corrected=args.apply_train_view,
            image_size=args.image_size,
            bbox_scale=args.bbox_scale,
        )
        baked = "R_mean+K_mean+t_mean" if args.apply_train_view else "K_mean+t_mean (R unchanged)"
        print(f"  wrote {out_path} ({baked}, train_view_corrected={args.apply_train_view})")
    else:
        print(
            "  note: camera not saved — re-run with --write-npz "
            "(use --apply-train-view for train/sanity camera)"
        )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "R_mean": R.tolist(),
            "K_mean": K.tolist(),
            "t_mean": t_opt.tolist(),
            "apply_train_view": args.apply_train_view,
            **info,
        }
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  report: {args.report_json}")


if __name__ == "__main__":
    main()
