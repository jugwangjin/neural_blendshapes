"""
Shape / integrity checks for face_expression_support + color_expression forward.

Run (from repo root, WSL):
  python debug/verify_face_expression_support_shapes.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _legacy_per_gaussian_support(ict, face_idx, alpha, dilate_rings, dtype, device):
    from model.blendshape_support import precompute_expression_support

    _, support = precompute_expression_support(ict, alpha=alpha, dilate_rings=dilate_rings)
    support = support.to(device=device, dtype=dtype)
    tri_vidx = ict.faces[face_idx.long()]
    return support[:, tri_vidx].amax(dim=-1).transpose(0, 1).contiguous()


def _simulate_densify_duplicate(surface):
    """Mirror duplicate_surf face_idx / color_expression resize (no optimizer)."""
    sel = torch.arange(min(8, surface.n_gaussians), device=surface.face_idx.device)
    n_new = sel.numel()
    n_old = surface.n_gaussians
    surface.register_buffer("face_idx", torch.cat([surface.face_idx, surface.face_idx[sel]]))
    p = surface.color_expression
    surface.color_expression = nn.Parameter(torch.cat([p.data, p.data[sel]], dim=0))
    return n_old, n_new


def _run_forward_color_expr(surface, verts, faces, k, device):
    expr = torch.linspace(0.0, 0.5, k, device=device, dtype=surface.color.dtype)
    out = surface._forward_surface(
        verts,
        faces,
        expr_coeff=expr,
        enable_color_pose=False,
        enable_color_expression=True,
    )
    assert out["color"].shape == (surface.n_gaussians, 3)
    assert torch.isfinite(out["color"]).all()
    return out


def main():
    from config import Config
    from model.gaussian_avatar import GaussianAvatar
    from model.ict_deformer import ICTDeformer
    from model.ict_model import ICTFaceKitTorch
    from model.tracker_mlp import TrackerCorrectionMLP

    cfg = Config()
    device = torch.device("cpu")
    ict = ICTFaceKitTorch(npy_dir=str(cfg.ict_npy))
    deformer = ICTDeformer(ict)
    avatar = GaussianAvatar.from_ict(ict, deformer=deformer, k_face=6, k_eyeball_sclera=4, k_eye_occlusion=4)
    surface = avatar.surface
    faces = ict.faces
    verts = ict.neutral_mesh[0] if ict.neutral_mesh.ndim == 3 else ict.neutral_mesh
    k = int(ict.num_expression)

    print("=== init ===")
    info = surface.validate_color_expression_shapes()
    print(info)

    legacy = _legacy_per_gaussian_support(
        ict,
        surface.face_idx,
        alpha=0.1,
        dilate_rings=4,
        dtype=surface.color.dtype,
        device=surface.face_idx.device,
    )
    gathered = surface._color_expression_support()
    err = (legacy - gathered).abs().max().item()
    assert err == 0.0, f"legacy [N,K] vs face_idx gather max err {err}"
    print(f"OK: legacy per-Gaussian support matches gather (max err={err})")

    _run_forward_color_expr(surface, verts, faces, k, device)
    print("OK: _forward_surface color_expression path")

    print("=== after simulated densify duplicate ===")
    n_old, n_new = _simulate_densify_duplicate(surface)
    assert surface.n_gaussians == n_old + n_new
    surface.validate_color_expression_shapes()
    _run_forward_color_expr(surface, verts, faces, k, device)
    print(f"OK: N {n_old} -> {surface.n_gaussians}, forward finite")

    print("=== after face_idx walk (random faces) ===")
    f = int(faces.shape[0])
    walk_idx = torch.arange(min(16, surface.n_gaussians), device=device)
    new_faces = torch.randint(0, f, (walk_idx.numel(),), device=device)
    surface.face_idx[walk_idx] = new_faces
    surface.validate_color_expression_shapes()
    for i, fi in zip(walk_idx.tolist(), new_faces.tolist()):
        row = surface._color_expression_support()[i]
        assert torch.equal(row, surface.face_expression_support[fi])
    _run_forward_color_expr(surface, verts, faces, k, device)
    print("OK: walked face_idx rows match face_expression_support[face]")

    print("=== tracker forward (enable_color_expression) ===")
    tracker = TrackerCorrectionMLP(
        n_blendshapes=cfg.num_mp_blendshapes,
        num_ict_expression=ict.num_expression,
        mediapipe_to_ict=ict.mediapipe_to_ict,
    )
    corr = tracker(
        mp_blendshape=torch.zeros(1, cfg.num_mp_blendshapes, device=device),
        mp_landmarks_2d=torch.zeros(1, 468, 2, device=device),
        mp_pose_raw=torch.zeros(1, 6, device=device),
        force_gamma_one=True,
    )
    out = avatar(
        tracker_out=corr,
        apply_expression_deform=True,
        enable_color_expression=True,
    )
    assert out["color"].shape[0] == surface.n_gaussians
    print(f"OK: avatar forward color {tuple(out['color'].shape)}")

    print("=== checkpoint_state load (legacy buffer stripped) ===")
    sd = avatar.state_dict()
    sd["color_expression_support"] = legacy[: min(legacy.shape[0], 3)]  # wrong shape on purpose
    from model.gaussian_avatar import GaussianAvatar as GA

    avatar2 = GA.from_checkpoint_state(ict, deformer, sd)
    s2 = avatar2.surface
    assert "face_expression_support" in s2._buffers
    assert s2.face_expression_support.shape == (
        int(ict.faces.shape[0]),
        int(ict.num_expression),
    )
    assert s2.n_gaussians == sd["face_idx"].shape[0]
    s2.validate_color_expression_shapes()
    print("OK: from_checkpoint_state recomputes face_expression_support")

    print("\nAll face_expression_support integrity checks passed.")


if __name__ == "__main__":
    main()
