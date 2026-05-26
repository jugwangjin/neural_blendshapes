"""NICP: staged ICT face fit to FLAME — inner 68[17:]+PIE jawline[0:16] + MP-free jaw KNN."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d import ops
from pytorch3d.loss import chamfer_distance

from processing.ict_flame_similarity import (
    expression_modes_as_vertex_deltas,
    landmark_match_indices_for_nicp,
)
from processing.ict_landmarks import landmark_jawline_vertex_indices
from processing.ict_mediapipe_lmk.constants import ICT_FACE_VERTEX_END
from processing.ict_mediapipe_lmk.landmarks import sample_bary
from processing.ict_mediapipe_lmk.nicp_template import apply_nicp_extension_to_full_mesh


def _add_large_steps_to_path(large_steps_root: Path):
    root = Path(large_steps_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def extract_face_patch(vertices, faces, vertex_end=ICT_FACE_VERTEX_END):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v_face = vertices[:vertex_end].copy()
    f_face = faces[np.all(faces < vertex_end, axis=1)]
    return v_face, f_face


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


def _prepare_landmark_targets(
    v_flame,
    f_flame,
    flame_lmk_face_idx,
    flame_lmk_bary,
    ict_landmark_indices,
    landmark_start,
):
    p_flame_all = sample_bary(v_flame, f_flame, flame_lmk_face_idx, flame_lmk_bary)
    match_idx = landmark_match_indices_for_nicp(
        ict_landmark_indices,
        len(p_flame_all),
        landmark_start=landmark_start,
        max_vertex_index=ICT_FACE_VERTEX_END,
    )
    if len(p_flame_all) >= 68:
        flame_p = np.asarray(p_flame_all, dtype=np.float64)[
            landmark_start : landmark_start + len(match_idx)
        ]
    else:
        flame_p = np.asarray(p_flame_all, dtype=np.float64)[: len(match_idx)]

    jaw_idx = landmark_jawline_vertex_indices(ict_landmark_indices, landmark_start)
    jaw_idx = jaw_idx[jaw_idx < ICT_FACE_VERTEX_END]
    print(
        f"  NICP inner: FLAME {len(p_flame_all)} pts, paired {len(flame_p)} "
        f"(landmark_start={landmark_start})"
    )
    print(f"  NICP jawline: {len(jaw_idx)} pts (Multi-PIE 0:{landmark_start}, no FLAME embedding)")
    return match_idx, flame_p, jaw_idx


def _as_pointcloud_batch(pts):
    """``(P,3)``, ``(1,P,3)``, or flat ``(P*3,)`` → ``(1,P,3)`` for pytorch3d ``knn_points``."""
    pts = pts.reshape(-1, 3)
    return pts.unsqueeze(0)


def _jaw_knn_loss(v_al, v_flame_t, jaw_idx_t, w_jaw):
    if w_jaw <= 0 or jaw_idx_t.numel() == 0:
        return torch.tensor(0.0, device=v_al.device, dtype=v_al.dtype)
    jaw_pts = _as_pointcloud_batch(v_al[jaw_idx_t])
    flame_pts = _as_pointcloud_batch(v_flame_t)
    if jaw_pts.shape[1] == 0:
        return torch.tensor(0.0, device=v_al.device, dtype=v_al.dtype)
    dists, _, _ = ops.knn_points(jaw_pts, flame_pts, K=1)
    return dists.sqrt().mean() * w_jaw


def _ict_modes_from_npy(ict_npy_dict, n_verts):
    neutral = np.asarray(ict_npy_dict["neutral_mesh"], dtype=np.float64)[:n_verts]
    exp_modes = expression_modes_as_vertex_deltas(
        ict_npy_dict["expression_shape_modes"], n_verts
    )
    idt_modes = expression_modes_as_vertex_deltas(
        ict_npy_dict["identity_shape_modes"], n_verts
    )
    names = list(ict_npy_dict["expression_names"])
    return neutral, exp_modes, idt_modes, names


def _eval_ict_verts(neutral_t, exp_modes_t, idt_modes_t, exp_w, idt_w):
    v = neutral_t + torch.einsum("e,evc->vc", exp_w, exp_modes_t)
    v = v + torch.einsum("i,ivc->vc", idt_w, idt_modes_t)
    return v


def _staged_blendshape_nicp(
    ict_npy_dict,
    v_flame,
    f_flame,
    match_idx,
    flame_lmk_t,
    jaw_idx,
    device,
    *,
    stage1_iters,
    stage1_lr,
    stage2_iters,
    stage2_lr,
    w68,
    w_jaw,
    w_chamfer,
    w_idt_reg,
    jaw_init,
):
    n_verts = ICT_FACE_VERTEX_END
    neutral, exp_modes, idt_modes, names = _ict_modes_from_npy(ict_npy_dict, len(ict_npy_dict["neutral_mesh"]))
    jaw_idx = names.index("jawOpen")
    n_exp = exp_modes.shape[0]
    n_idt = idt_modes.shape[0]

    neutral_t = torch.tensor(neutral[:n_verts], dtype=torch.float32, device=device)
    exp_modes_t = torch.tensor(exp_modes[:, :n_verts], dtype=torch.float32, device=device)
    idt_modes_t = torch.tensor(idt_modes[:, :n_verts], dtype=torch.float32, device=device)
    match_idx_t = torch.tensor(match_idx, dtype=torch.long, device=device)
    jaw_idx_t = torch.tensor(jaw_idx, dtype=torch.long, device=device)

    v_flame_t = torch.tensor(v_flame, dtype=torch.float32, device=device).reshape(-1, 3)

    jaw = torch.nn.Parameter(torch.tensor(float(jaw_init), device=device))
    zero_exp = torch.zeros(n_exp, device=device)
    zero_idt = torch.zeros(n_idt, device=device)

    opt1 = torch.optim.Adam([jaw], lr=stage1_lr)
    print(f"  NICP stage1: jaw + s,R,T ({stage1_iters} iters)")
    for _ in range(stage1_iters):
        opt1.zero_grad()
        exp_w = zero_exp.clone()
        exp_w[jaw_idx] = jaw
        v_face = _eval_ict_verts(neutral_t, exp_modes_t, idt_modes_t, exp_w, zero_idt)
        ict_lmk = v_face[match_idx_t]
        align = ops.corresponding_points_alignment(
            ict_lmk.unsqueeze(0), flame_lmk_t.unsqueeze(0), estimate_scale=True
        )
        R = align.R[0]
        v_al = align.s[0] * (v_face @ R) + align.T[0]
        lmk_loss = (align.s[0] * (ict_lmk @ R) + align.T[0] - flame_lmk_t).abs().mean() * w68
        ch_loss, _ = chamfer_distance(v_al.unsqueeze(0), v_flame_t.unsqueeze(0), single_directional=True)
        jaw_loss = _jaw_knn_loss(v_al, v_flame_t, jaw_idx_t, w_jaw)
        loss = lmk_loss + ch_loss * w_chamfer + jaw_loss
        loss.backward()
        opt1.step()

    idt_w = torch.nn.Parameter(torch.zeros(n_idt, device=device))
    opt2 = torch.optim.Adam([jaw, idt_w], lr=stage2_lr)
    print(f"  NICP stage2: jaw + identity + s,R,T ({stage2_iters} iters)")
    for _ in range(stage2_iters):
        opt2.zero_grad()
        exp_w = zero_exp.clone()
        exp_w[jaw_idx] = jaw
        v_face = _eval_ict_verts(neutral_t, exp_modes_t, idt_modes_t, exp_w, idt_w)
        ict_lmk = v_face[match_idx_t]
        align = ops.corresponding_points_alignment(
            ict_lmk.unsqueeze(0), flame_lmk_t.unsqueeze(0), estimate_scale=True
        )
        R = align.R[0]
        v_al = align.s[0] * (v_face @ R) + align.T[0]
        lmk_loss = (align.s[0] * (ict_lmk @ R) + align.T[0] - flame_lmk_t).abs().mean() * w68
        ch_loss, _ = chamfer_distance(v_al.unsqueeze(0), v_flame_t.unsqueeze(0), single_directional=True)
        jaw_loss = _jaw_knn_loss(v_al, v_flame_t, jaw_idx_t, w_jaw)
        loss = lmk_loss + ch_loss * w_chamfer + jaw_loss + (idt_w**2).mean() * w_idt_reg
        loss.backward()
        opt2.step()

    with torch.no_grad():
        exp_w = zero_exp.clone()
        exp_w[jaw_idx] = jaw
        v_face = _eval_ict_verts(neutral_t, exp_modes_t, idt_modes_t, exp_w, idt_w)
        align = ops.corresponding_points_alignment(
            v_face[match_idx_t].unsqueeze(0), flame_lmk_t.unsqueeze(0), estimate_scale=True
        )
        R = align.R[0]
        v_face = align.s[0] * (v_face @ R) + align.T[0]
    return v_face.detach().cpu().numpy()


def _vertex_residual_nicp(
    v_face_init,
    f_face,
    v_flame,
    f_flame,
    match_idx,
    flame_lmk_t,
    jaw_idx,
    large_steps_root,
    device,
    *,
    iterations,
    lr,
    lambda_large_steps,
    w68,
    w_jaw,
    wsurf,
    w_chamfer,
    wnormal,
    wedge,
):
    _add_large_steps_to_path(large_steps_root)
    from largesteps.geometry import compute_matrix
    from largesteps.parameterize import from_differential, to_differential

    v_flame_t = torch.tensor(v_flame, dtype=torch.float32, device=device).reshape(-1, 3)
    f_flame_t = torch.tensor(f_flame, dtype=torch.long, device=device)
    flame_vn = _vertex_normals(v_flame_t, f_flame_t)
    match_idx_t = torch.tensor(match_idx, dtype=torch.long, device=device)
    jaw_idx_t = torch.tensor(jaw_idx, dtype=torch.long, device=device)

    v0 = torch.tensor(v_face_init, dtype=torch.float32, device=device)
    f_t = torch.tensor(f_face, dtype=torch.long, device=device)
    e0 = _edge_lengths(v0, f_t)

    m_mat = compute_matrix(v0, f_t, lambda_=lambda_large_steps)
    u = torch.nn.Parameter(to_differential(m_mat, v0))
    opt = torch.optim.Adam([u], lr=lr)

    print(
        f"  NICP stage3: Large Steps + inner lmk + jaw KNN "
        f"({iterations} iters, λ_ls={lambda_large_steps})"
    )
    for _ in range(iterations):
        opt.zero_grad()
        v_def = from_differential(m_mat, u)

        lmk_loss = (v_def[match_idx_t] - flame_lmk_t).abs().mean() * w68
        jaw_loss = _jaw_knn_loss(v_def, v_flame_t, jaw_idx_t, w_jaw)

        dists, idx_nn, _ = ops.knn_points(
            _as_pointcloud_batch(v_def), _as_pointcloud_batch(v_flame_t), K=1
        )
        surf_loss = dists.sqrt().mean() * wsurf

        ch_loss, _ = chamfer_distance(
            v_def.unsqueeze(0), v_flame_t.unsqueeze(0), single_directional=False
        )
        ch_loss = ch_loss * w_chamfer

        n_def = _vertex_normals(v_def, f_t)
        n_tgt = flame_vn[idx_nn.squeeze(0).squeeze(-1)]
        normal_loss = (1.0 - (n_def * n_tgt).sum(dim=1).clamp(-1, 1)).mean() * wnormal

        e1 = _edge_lengths(v_def, f_t)
        edge_loss = sum((a - b).abs().mean() for a, b in zip(e1, e0)) / 3.0 * wedge

        loss = lmk_loss + jaw_loss + surf_loss + ch_loss + normal_loss + edge_loss
        loss.backward()
        opt.step()

    return from_differential(m_mat, u).detach().cpu().numpy()


def _legacy_vertex_nicp(
    v_face_init,
    f_face,
    v_flame,
    f_flame,
    match_idx,
    flame_lmk_t,
    jaw_idx,
    large_steps_root,
    device,
    *,
    iterations,
    lr,
    lambda_large_steps,
    w68,
    w_jaw,
    wsurf,
    w_chamfer,
    wnormal,
    wedge,
    skip_rigid_init,
    p_ict_match,
    p_flame_match,
):
    if not skip_rigid_init:
        _, scale, r, t = rigid_align_paired(p_ict_match, p_flame_match)
        v_face_init = apply_similarity(v_face_init, scale, r, t)

    return _vertex_residual_nicp(
        v_face_init,
        f_face,
        v_flame,
        f_flame,
        match_idx,
        flame_lmk_t,
        jaw_idx,
        large_steps_root,
        device,
        iterations=iterations,
        lr=lr,
        lambda_large_steps=lambda_large_steps,
        w68=w68,
        w_jaw=w_jaw,
        wsurf=wsurf,
        w_chamfer=w_chamfer,
        wnormal=wnormal,
        wedge=wedge,
    )


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
    w_jaw=30.0,
    wsurf=1.0,
    w_chamfer_bidir=0.25,
    wnormal=0.1,
    wedge=1.0,
    landmark_start=17,
    skip_rigid_init=False,
    ict_npy_dict=None,
    stage1_iters=150,
    stage1_lr=5e-3,
    stage2_iters=400,
    stage2_lr=1e-2,
    stage3_iters=None,
    w_idt_reg=0.05,
    jaw_init=None,
    regions=None,
    propagate_extension=True,
    extension_iters=12,
):
    v_flame = np.asarray(v_flame, dtype=np.float64)
    f_flame = np.asarray(f_flame, dtype=np.int64)

    match_idx, flame_lmk_np, jaw_idx = _prepare_landmark_targets(
        v_flame,
        f_flame,
        flame_lmk_face_idx,
        flame_lmk_bary,
        ict_landmark_indices,
        landmark_start,
    )
    flame_lmk_t = torch.tensor(flame_lmk_np, dtype=torch.float32, device=device)

    v_face_init, f_face = extract_face_patch(v_ict_full, f_ict_full)

    if ict_npy_dict is not None:
        s3_iters = iterations if stage3_iters is None else int(stage3_iters)
        if jaw_init is None:
            jaw_init = float(ict_npy_dict.get("flame_similarity_ict_jaw_open", 0.75))
        v_face_bs = _staged_blendshape_nicp(
            ict_npy_dict,
            v_flame,
            f_flame,
            match_idx,
            flame_lmk_t,
            jaw_idx,
            device,
            stage1_iters=stage1_iters,
            stage1_lr=stage1_lr,
            stage2_iters=stage2_iters,
            stage2_lr=stage2_lr,
            w68=w68,
            w_jaw=w_jaw,
            w_chamfer=w_chamfer_bidir,
            w_idt_reg=w_idt_reg,
            jaw_init=jaw_init,
        )
        if s3_iters > 0:
            v_face_fit = _vertex_residual_nicp(
                v_face_bs,
                f_face,
                v_flame,
                f_flame,
                match_idx,
                flame_lmk_t,
                jaw_idx,
                large_steps_root,
                device,
                iterations=s3_iters,
                lr=lr * 0.5,
                lambda_large_steps=lambda_large_steps * 3.0,
                w68=w68,
                w_jaw=w_jaw,
                wsurf=wsurf * 0.25,
                w_chamfer=w_chamfer_bidir * 0.15,
                wnormal=wnormal * 0.5,
                wedge=wedge * 2.0,
            )
        else:
            v_face_fit = v_face_bs
    else:
        ict_pts = v_face_init[match_idx]
        v_face_fit = _legacy_vertex_nicp(
            v_face_init,
            f_face,
            v_flame,
            f_flame,
            match_idx,
            flame_lmk_t,
            jaw_idx,
            large_steps_root,
            device,
            iterations=iterations,
            lr=lr,
            lambda_large_steps=lambda_large_steps,
            w68=w68,
            w_jaw=w_jaw,
            wsurf=wsurf,
            w_chamfer=w_chamfer_bidir,
            wnormal=wnormal,
            wedge=wedge,
            skip_rigid_init=skip_rigid_init,
            p_ict_match=ict_pts,
            p_flame_match=flame_lmk_np,
        )

    v_ict_fit = np.asarray(v_ict_full, dtype=np.float64).copy()
    v_ict_fit[:ICT_FACE_VERTEX_END] = v_face_fit
    if propagate_extension and regions is not None:
        v_ict_fit = apply_nicp_extension_to_full_mesh(
            v_ict_full,
            v_ict_fit,
            f_ict_full,
            regions,
            n_iters=extension_iters,
        )
        from processing.ict_mediapipe_lmk.nicp_template import nicp_extension_vertex_indices

        n_ext = len(nicp_extension_vertex_indices(regions))
        print(f"  NICP extension: propagated displacement to {n_ext} verts (mouth/eye-socket/occlusion)")
    return v_ict_fit, v_face_fit


def estimate_similarity(source, target):
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
