"""NICP: fit ICT face patch to FLAME canonical using Large Steps parameterization."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d import ops

from ict_mediapipe_lmk.constants import ICT_FACE_FACE_END, ICT_FACE_VERTEX_END
from ict_mediapipe_lmk.landmarks import sample_bary


def _add_large_steps_to_path(large_steps_root: Path):
    root = Path(large_steps_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def extract_face_patch(vertices, faces):
    v_face = vertices[:ICT_FACE_VERTEX_END].copy()
    f_face = faces[:ICT_FACE_FACE_END].copy()
    return v_face, f_face


def estimate_similarity(source, target):
    """Similarity transform from paired 3D points (K, 3)."""
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean
    h = src_c.T @ tgt_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    scale = np.trace(tgt_c.T @ (src_c @ r)) / (np.trace(src_c.T @ src_c) + 1e-8)
    t = tgt_mean - scale * (src_mean @ r)
    return scale, r, t


def apply_similarity(vertices, scale, rotation, translation):
    return scale * (vertices @ rotation) + translation


def rigid_align_paired(source, target):
    scale, r, t = estimate_similarity(source, target)
    return apply_similarity(source, scale, r, t), scale, r, t


def _edge_lengths(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return (
        (v1 - v0).norm(dim=1),
        (v2 - v1).norm(dim=1),
        (v0 - v2).norm(dim=1),
    )


def _face_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    return F.normalize(n, dim=1)


def _vertex_normals(verts, faces):
    fn = _face_normals(verts, faces)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    return F.normalize(vn, dim=1)


def fit_ict_face_to_flame(
    v_ict_full,
    f_ict_full,
    v_flame,
    f_flame,
    ict_landmark_indices,
    flame_lmk_face_idx,
    flame_lmk_bary,
    large_steps_root,
    device,
    iterations=300,
    lr=1e-2,
    lambda_large_steps=10.0,
    w68=100.0,
    wsurf=1.0,
    wnormal=0.1,
    wedge=1.0,
    landmark_start=17,
):
    _add_large_steps_to_path(large_steps_root)
    from largesteps.geometry import compute_matrix
    from largesteps.parameterize import from_differential, to_differential

    v_face_init, f_face = extract_face_patch(
        np.asarray(v_ict_full, dtype=np.float64),
        np.asarray(f_ict_full, dtype=np.int64),
    )

    v_flame = np.asarray(v_flame, dtype=np.float64)
    f_flame = np.asarray(f_flame, dtype=np.int64)

    p_flame_68 = sample_bary(
        v_flame,
        f_flame,
        flame_lmk_face_idx,
        flame_lmk_bary,
    )
    ict_lmk_idx = np.asarray(ict_landmark_indices, dtype=np.int64)
    n_match = min(len(p_flame_68), len(ict_lmk_idx)) - landmark_start
    match_idx = ict_lmk_idx[landmark_start : landmark_start + n_match]
    face_mask = match_idx < ICT_FACE_VERTEX_END
    match_idx = match_idx[face_mask]
    p_flame_match = p_flame_68[landmark_start : landmark_start + n_match][face_mask]
    p_ict_match = v_face_init[match_idx]

    _, scale, r, t = rigid_align_paired(p_ict_match, p_flame_match)
    v_face_init = apply_similarity(v_face_init, scale, r, t)

    v_flame_t = torch.tensor(v_flame, dtype=torch.float32, device=device)
    f_flame_t = torch.tensor(f_flame, dtype=torch.long, device=device)
    flame_vn = _vertex_normals(v_flame_t, f_flame_t)

    p_flame_match_t = torch.tensor(p_flame_match, dtype=torch.float32, device=device)

    v0 = torch.tensor(v_face_init, dtype=torch.float32, device=device)
    f_t = torch.tensor(f_face, dtype=torch.long, device=device)
    e0 = _edge_lengths(v0, f_t)

    m_mat = compute_matrix(v0, f_t, lambda_=lambda_large_steps)
    u = torch.nn.Parameter(to_differential(m_mat, v0))
    opt = torch.optim.Adam([u], lr=lr)

    for _ in range(iterations):
        opt.zero_grad()
        v_def = from_differential(m_mat, u)

        lmk_loss = (v_def[match_idx] - p_flame_match_t).abs().mean() * w68

        dists, idx_nn, _ = ops.knn_points(
            v_def.unsqueeze(0),
            v_flame_t.unsqueeze(0),
            K=1,
        )
        surf_loss = dists.sqrt().mean() * wsurf

        n_def = _vertex_normals(v_def, f_t)
        n_tgt = flame_vn[idx_nn.squeeze(0).squeeze(-1)]
        normal_loss = (1.0 - (n_def * n_tgt).sum(dim=1).clamp(-1, 1)).mean() * wnormal

        e1 = _edge_lengths(v_def, f_t)
        edge_loss = sum((a - b).abs().mean() for a, b in zip(e1, e0)) / 3.0 * wedge

        loss = lmk_loss + surf_loss + normal_loss + edge_loss
        loss.backward()
        opt.step()

    v_face_fit = from_differential(m_mat, u).detach().cpu().numpy()
    v_ict_fit = np.asarray(v_ict_full, dtype=np.float64).copy()
    v_ict_fit[:ICT_FACE_VERTEX_END] = v_face_fit
    return v_ict_fit, v_face_fit
