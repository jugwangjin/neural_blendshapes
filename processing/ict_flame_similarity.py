"""
ICT ↔ FLAME alignment: coarse ``s,T`` + optimized ``jawOpen`` (npy build), then ``s,R,T`` (bake).

Local ICT mesh → FLAME space: ``x_flame = s * (x @ R) + T`` (row-vector convention).

Coarse (no R): fit on landmarks after ``neutral + jawOpen * w``.
Bake rigid: ``corresponding_points_alignment`` on coarse-aligned landmarks; composed into
``flame_alignment_s`` / ``flame_alignment_R`` / ``flame_alignment_T``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from processing.ict_landmarks import (
    LANDMARK_START_FLAME_PAIRING,
    landmark_jawline_vertex_indices,
)
from processing.paths import FLAME_MODEL, FLAME_STATIC_EMBEDDING


def expression_modes_as_vertex_deltas(modes, n_verts):
    """(n_exp, V, 3) or (n_exp, V*3) → (n_exp, n_verts, 3)."""
    m = np.asarray(modes, dtype=np.float64)
    if m.ndim == 3:
        return m[:, :n_verts, :]
    if m.ndim != 2:
        raise ValueError(f"expression_shape_modes: expected 2D or 3D, got shape {m.shape}")
    n_exp, dim = m.shape
    if dim == n_verts * 3:
        return m.reshape(n_exp, n_verts, 3)
    if dim == n_verts:
        return m[:, :n_verts, None]
    return m.reshape(n_exp, -1, 3)[:, :n_verts, :]


def load_flame_static_embedding(path=None):
    path = Path(path or FLAME_STATIC_EMBEDDING)
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return (
        np.asarray(data["lmk_face_idx"], dtype=np.int64),
        np.asarray(data["lmk_b_coords"], dtype=np.float64),
    )


def sample_landmarks_bary(vertices, faces, face_idx, bary):
    tri = faces[face_idx]
    pts = vertices[tri]
    return (pts * bary[:, :, None]).sum(axis=1)


def ict_neutral_with_jaw_open(vertices, expression_shape_modes, expression_names, jaw_open=0.75):
    names = list(expression_names)
    jaw_idx = names.index("jawOpen")
    n_verts = len(vertices)
    modes = expression_modes_as_vertex_deltas(expression_shape_modes, n_verts)
    return np.asarray(vertices, dtype=np.float64) + modes[jaw_idx] * float(jaw_open)


def load_flame_neutral_vertices(
    flame_model_path=None,
    *,
    use_processed_faces=False,
    use_canonical_pose=True,
    device="cpu",
):
    from flame.FLAME import FLAME

    device = torch.device(device)
    shape = torch.zeros(1, 100)
    flame = FLAME(
        str(flame_model_path or FLAME_MODEL),
        n_shape=100,
        n_exp=50,
        shape_params=shape,
        use_processed_faces=use_processed_faces,
    ).to(device)
    exp = torch.zeros(1, flame.n_exp, device=device)
    if use_canonical_pose:
        pose = flame.canonical_pose.to(device)
    else:
        pose = torch.zeros(1, 15, device=device)
    verts, _, _ = flame(expression_params=exp, full_pose=pose)
    faces = flame.faces_tensor.cpu().numpy()
    return verts[0].cpu().numpy(), faces


def flame_landmarks_68(
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    *,
    use_processed_faces=False,
    use_canonical_pose=True,
    device="cpu",
):
    verts, faces = load_flame_neutral_vertices(
        flame_model_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )
    face_idx, bary = load_flame_static_embedding(flame_lmk_embedding_path)
    pts = sample_landmarks_bary(verts, faces, face_idx, bary)
    return pts, face_idx, bary


def ict_landmarks_from_vertices(vertices, landmark_indices, landmark_start=17):
    idx = np.asarray(landmark_indices, dtype=np.int64)[landmark_start:]
    return np.asarray(vertices, dtype=np.float64)[idx]


def ict_jawline_from_vertices(vertices, landmark_indices, landmark_start=LANDMARK_START_FLAME_PAIRING):
    idx = landmark_jawline_vertex_indices(landmark_indices, landmark_start)
    return np.asarray(vertices, dtype=np.float64)[idx]


def jawline_knn_mean(ict_jaw_pts, flame_verts):
    """Mean min-distance from each ICT jawline point to FLAME mesh (no FLAME jaw embedding)."""
    ict_jaw_pts = np.asarray(ict_jaw_pts, dtype=np.float64)
    flame_verts = np.asarray(flame_verts, dtype=np.float64)
    if ict_jaw_pts.size == 0 or flame_verts.size == 0:
        return 0.0
    d = np.linalg.norm(ict_jaw_pts[:, None, :] - flame_verts[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def fit_uniform_scale_translation(source, target):
    """``target ≈ s * source + T`` with scalar ``s`` (uniform over x,y,z)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src = source - mu_s
    tgt = target - mu_t
    den = np.sum(src * src)
    s = 1.0 if den < 1e-12 else float(np.sum(src * tgt) / den)
    t = mu_t - s * mu_s
    return s, t


def pair_landmarks_for_alignment(
    flame_pts, ict_pts, flame_landmark_start=LANDMARK_START_FLAME_PAIRING
):
    """
    ``ict_pts`` = ``landmark_indices[landmark_start:]`` (typically 51).

    FLAME embedding: 68 Multi-PIE → slice ``[17:]``; ~51 inner-only → use all (no slice).
    Slicing 51 pts at 17 yields 34 pairs and breaks alignment.
    """
    flame_pts = np.asarray(flame_pts, dtype=np.float64)
    ict_pts = np.asarray(ict_pts, dtype=np.float64)
    if len(flame_pts) >= 68:
        flame_pts = flame_pts[flame_landmark_start:]
    n = min(len(flame_pts), len(ict_pts))
    if n < 3:
        raise ValueError(f"Need >= 3 landmark pairs for alignment, got {n}")
    return flame_pts[:n], ict_pts[:n]


def landmark_match_indices_for_nicp(
    ict_landmark_indices,
    n_flame_landmarks,
    *,
    landmark_start=LANDMARK_START_FLAME_PAIRING,
    max_vertex_index=None,
):
    """
    ICT vertex indices for NICP / 68-lmk loss, paired with FLAME embedding count.

    Same pairing rules as ``pair_landmarks_for_alignment`` (68→``[17:]``, 51→all).
    """
    ict_idx = np.asarray(ict_landmark_indices, dtype=np.int64)
    ict_inner = ict_idx[landmark_start:]
    if n_flame_landmarks >= 68:
        n_pairs = min(n_flame_landmarks - landmark_start, len(ict_inner))
    else:
        n_pairs = min(n_flame_landmarks, len(ict_inner))
    match_idx = ict_inner[:n_pairs]
    if max_vertex_index is not None:
        match_idx = match_idx[match_idx < max_vertex_index]
    return match_idx


def fit_similarity_rigid(source, target, device="cpu"):
    """``target ≈ s * (source @ R) + T`` via pytorch3d (rotation + uniform scale)."""
    from pytorch3d import ops

    device = torch.device(device)
    src = torch.tensor(np.asarray(source, dtype=np.float64)[None], dtype=torch.float32, device=device)
    tgt = torch.tensor(np.asarray(target, dtype=np.float64)[None], dtype=torch.float32, device=device)
    out = ops.corresponding_points_alignment(src, tgt, estimate_scale=True)
    # pytorch3d: s shape (B,), R (B,3,3), T (B,3) — not (B,1,1)
    s = float(out.s.detach().cpu().reshape(-1)[0])
    R = out.R.detach().cpu().numpy().astype(np.float64)
    if R.ndim == 3:
        R = R[0]
    R = R.reshape(3, 3)
    T = out.T.detach().cpu().numpy().astype(np.float64)
    if T.ndim == 2:
        T = T[0]
    T = T.reshape(3)
    return s, R, T


def compose_st_then_similarity(s0, t0, s1, R1, t1):
    """
    Apply coarse ``x' = s0*x + t0`` then rigid ``x'' = s1*(x' @ R1) + t1`` as single ``s,R,T`` on ``x``.
    """
    t0 = np.asarray(t0, dtype=np.float64).reshape(3)
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    t1 = np.asarray(t1, dtype=np.float64).reshape(3)
    s = float(s0) * float(s1)
    R = R1
    T = float(s1) * (t0 @ R1) + t1
    return s, R, T


def apply_flame_similarity(vertices, s, t):
    """Coarse: ``x' = s * x + T``."""
    v = np.asarray(vertices, dtype=np.float64)
    return float(s) * v + np.asarray(t, dtype=np.float64).reshape(3)


def apply_flame_alignment(vertices, s, R, t):
    """Full: ``x' = s * (x @ R) + T``."""
    v = np.asarray(vertices, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return float(s) * (v @ R) + t


def flame_similarity_from_npy(model_dict):
    s = float(model_dict.get("flame_similarity_s", 1.0))
    t = np.asarray(model_dict.get("flame_similarity_T", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return s, t


def flame_alignment_from_npy(model_dict):
    """Final composed ICT → FLAME transform (after bake)."""
    if "flame_alignment_s" not in model_dict:
        return None
    s = float(model_dict["flame_alignment_s"])
    R = np.asarray(model_dict["flame_alignment_R"], dtype=np.float64).reshape(3, 3)
    T = np.asarray(model_dict["flame_alignment_T"], dtype=np.float64).reshape(3)
    return s, R, T


def has_flame_alignment(model_dict):
    return "flame_alignment_s" in model_dict and "flame_alignment_R" in model_dict


def ict_mesh_with_jaw(vertices, model_dict):
    jaw = float(model_dict.get("flame_similarity_ict_jaw_open", 0.75))
    return ict_neutral_with_jaw_open(
        vertices,
        model_dict["expression_shape_modes"],
        model_dict["expression_names"],
        jaw_open=jaw,
    )


def ict_mesh_coarse_aligned(vertices, model_dict):
    """``neutral + jawOpen*w`` then coarse ``s,T`` (no R)."""
    v = ict_mesh_with_jaw(vertices, model_dict)
    s, t = flame_similarity_from_npy(model_dict)
    return apply_flame_similarity(v, s, t)


def apply_ict_to_flame_space(vertices, model_dict, *, use_final_alignment=True):
    """
    Map ICT vertices to FLAME space using npy fields.

    Uses ``flame_alignment_*`` when present and ``use_final_alignment``; else coarse ``s,T`` after jaw.
    """
    v = ict_mesh_with_jaw(vertices, model_dict)
    if use_final_alignment:
        aligned = flame_alignment_from_npy(model_dict)
        if aligned is not None:
            s, R, T = aligned
            return apply_flame_alignment(v, s, R, T)
    s, t = flame_similarity_from_npy(model_dict)
    return apply_flame_similarity(v, s, t)


def landmark_pair_count(
    flame_pts, landmark_indices, landmark_start=LANDMARK_START_FLAME_PAIRING
):
    flame_pts = np.asarray(flame_pts, dtype=np.float64)
    ict_n = len(np.asarray(landmark_indices, dtype=np.int64)[landmark_start:])
    if len(flame_pts) >= 68:
        return min(len(flame_pts) - landmark_start, ict_n)
    return min(len(flame_pts), ict_n)


def print_flame_alignment_report(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    model_dict,
    *,
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    landmark_start=LANDMARK_START_FLAME_PAIRING,
    device="cpu",
):
    """Print landmark RMSE before/after jaw+coarse and after final alignment (if present)."""
    use_proc = bool(model_dict.get("flame_similarity_use_processed_faces", False))
    use_pose = bool(model_dict.get("flame_similarity_use_canonical_pose", True))
    flame_pts, _, _ = flame_landmarks_68(
        flame_model_path,
        flame_lmk_embedding_path,
        use_processed_faces=use_proc,
        use_canonical_pose=use_pose,
        device=device,
    )
    jaw = float(model_dict.get("flame_similarity_ict_jaw_open", 0.75))
    ict_jaw = ict_neutral_with_jaw_open(
        ict_vertices, expression_shape_modes, expression_names, jaw_open=jaw
    )
    ict_raw = ict_landmarks_from_vertices(ict_jaw, landmark_indices, landmark_start)
    flame_p, ict_raw_p = pair_landmarks_for_alignment(
        flame_pts, ict_raw, flame_landmark_start=landmark_start
    )
    err0 = np.linalg.norm(ict_raw_p - flame_p, axis=1)

    s0, t0 = flame_similarity_from_npy(model_dict)
    aligned_pack = flame_alignment_from_npy(model_dict)
    flame_verts, _ = load_flame_neutral_vertices(
        flame_model_path,
        use_processed_faces=use_proc,
        use_canonical_pose=use_pose,
        device=device,
    )
    ict_jaw_raw = ict_jawline_from_vertices(ict_jaw, landmark_indices, landmark_start)
    n_flame = len(flame_pts)
    flame_mode = f"68 slice [{landmark_start}:]" if n_flame >= 68 else f"{n_flame} inner (no slice)"
    lines = [
        f"  FLAME lmk count={n_flame} ({flame_mode}), paired n={len(flame_p)}",
        f"  jawline 0:{landmark_start}: n={len(ict_jaw_raw)} (MP bake has no jaw contour)",
        f"  jawOpen only (inner): mean={err0.mean():.6f} max={err0.max():.6f} (jawOpen={jaw:.4f})",
    ]
    if aligned_pack is not None:
        s, R, T = aligned_pack
        ict_fin = apply_flame_alignment(ict_raw_p, s, R, T)
        err_fin = np.linalg.norm(ict_fin - flame_p, axis=1)
        jaw_knn = jawline_knn_mean(apply_flame_alignment(ict_jaw_raw, s, R, T), flame_verts)
        lines.append(
            f"  after jaw+s,R,T: inner mean={err_fin.mean():.6f} max={err_fin.max():.6f} "
            f"jaw KNN={jaw_knn:.6f} (s={s:.6f})"
        )
    else:
        ict_coarse = apply_flame_similarity(ict_raw_p, s0, t0)
        err1 = np.linalg.norm(ict_coarse - flame_p, axis=1)
        jaw_knn = jawline_knn_mean(apply_flame_similarity(ict_jaw_raw, s0, t0), flame_verts)
        lines.append(
            f"  after jaw+s,T:  inner mean={err1.mean():.6f} max={err1.max():.6f} "
            f"jaw KNN={jaw_knn:.6f} (s={s0:.6f})"
        )
    print("\n".join(lines))


def _landmark_st_error(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    flame_pts,
    *,
    jaw_open,
    landmark_start=17,
):
    ict_verts = ict_neutral_with_jaw_open(
        ict_vertices, expression_shape_modes, expression_names, jaw_open=jaw_open
    )
    ict_pts = ict_landmarks_from_vertices(ict_verts, landmark_indices, landmark_start)
    flame_p, ict_p = pair_landmarks_for_alignment(flame_pts, ict_pts, flame_landmark_start=landmark_start)
    s, t = fit_uniform_scale_translation(ict_p, flame_p)
    aligned = apply_flame_similarity(ict_p, s, t)
    err = np.linalg.norm(aligned - flame_p, axis=1)
    return float(err.mean()), float(err.max()), s, t, float(jaw_open)


def _landmark_rigid_error(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    flame_pts,
    *,
    jaw_open,
    landmark_start=17,
    device="cpu",
    flame_verts=None,
    w_jaw_knn=0.0,
):
    ict_verts = ict_neutral_with_jaw_open(
        ict_vertices, expression_shape_modes, expression_names, jaw_open=jaw_open
    )
    ict_pts = ict_landmarks_from_vertices(ict_verts, landmark_indices, landmark_start)
    flame_p, ict_p = pair_landmarks_for_alignment(flame_pts, ict_pts, flame_landmark_start=landmark_start)
    s, R, T = fit_similarity_rigid(ict_p, flame_p, device=device)
    aligned = apply_flame_alignment(ict_p, s, R, T)
    err = np.linalg.norm(aligned - flame_p, axis=1)
    inner_mean = float(err.mean())
    inner_max = float(err.max())
    jaw_knn = 0.0
    if flame_verts is not None and w_jaw_knn > 0:
        ict_jaw = ict_jawline_from_vertices(ict_verts, landmark_indices, landmark_start)
        ict_jaw_al = apply_flame_alignment(ict_jaw, s, R, T)
        jaw_knn = jawline_knn_mean(ict_jaw_al, flame_verts)
    score_mean = inner_mean + float(w_jaw_knn) * jaw_knn
    return score_mean, inner_max, s, R, T, float(jaw_open), inner_mean, jaw_knn


def optimize_ict_jaw_open(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    flame_pts,
    *,
    landmark_start=17,
    jaw_min=0.0,
    jaw_max=1.2,
    n_jaw_steps=25,
    initial_jaw=0.75,
    use_rigid=True,
    device="cpu",
    flame_verts=None,
    w_jaw_knn=25.0,
):
    """Grid-search ``jawOpen``; score with rigid ``s,R,T`` (default) or coarse ``s,T``."""
    grid = np.linspace(float(jaw_min), float(jaw_max), int(n_jaw_steps))
    if initial_jaw is not None and initial_jaw not in grid:
        grid = np.sort(np.concatenate([grid, [float(initial_jaw)]]))

    best = None
    for jaw in grid:
        if use_rigid:
            score_e, max_e, s, R, T, j, inner_mean, jaw_knn = _landmark_rigid_error(
                ict_vertices,
                landmark_indices,
                expression_shape_modes,
                expression_names,
                flame_pts,
                jaw_open=jaw,
                landmark_start=landmark_start,
                device=device,
                flame_verts=flame_verts,
                w_jaw_knn=w_jaw_knn,
            )
            cand = (score_e, max_e, s, R, T, j, inner_mean, jaw_knn)
        else:
            mean_e, max_e, s, t, j = _landmark_st_error(
                ict_vertices,
                landmark_indices,
                expression_shape_modes,
                expression_names,
                flame_pts,
                jaw_open=jaw,
                landmark_start=landmark_start,
            )
            cand = (mean_e, max_e, s, None, t, j, mean_e, 0.0)

        if best is None or cand[0] < best[0]:
            best = cand

    score_e, max_e, s, R, T_or_t, jaw, inner_mean, jaw_knn = best
    out = {
        "flame_similarity_ict_jaw_open": float(jaw),
        "flame_similarity_lmk_err_mean": float(inner_mean),
        "flame_similarity_lmk_err_max": float(max_e),
        "flame_similarity_jaw_knn_mean": float(jaw_knn),
    }
    if use_rigid:
        out["flame_alignment_s"] = np.float32(s)
        out["flame_alignment_R"] = np.asarray(R, dtype=np.float32)
        out["flame_alignment_T"] = np.asarray(T_or_t, dtype=np.float32)
        out["flame_similarity_s"] = np.float32(1.0)
        out["flame_similarity_T"] = np.zeros(3, dtype=np.float32)
    else:
        out["flame_similarity_s"] = np.float32(s)
        out["flame_similarity_T"] = np.asarray(T_or_t, dtype=np.float32)
    return out


def default_flame_similarity_fields(
    *,
    landmark_start=17,
    use_processed_faces=False,
    use_canonical_pose=True,
    ict_jaw_open=0.75,
):
    return {
        "flame_similarity_s": np.float32(1.0),
        "flame_similarity_T": np.zeros(3, dtype=np.float32),
        "flame_similarity_landmark_start": int(landmark_start),
        "flame_similarity_n_pairs": np.int32(0),
        "flame_similarity_lmk_err_mean": np.float32(0.0),
        "flame_similarity_lmk_err_max": np.float32(0.0),
        "flame_similarity_use_processed_faces": bool(use_processed_faces),
        "flame_similarity_use_canonical_pose": bool(use_canonical_pose),
        "flame_similarity_ict_jaw_open": float(ict_jaw_open),
    }


def compute_ict_flame_similarity(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    *,
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    use_processed_faces=False,
    use_canonical_pose=True,
    landmark_start=17,
    ict_jaw_open=0.75,
    optimize_jaw=True,
    jaw_min=0.0,
    jaw_max=1.2,
    n_jaw_steps=25,
    device="cpu",
    w_jaw_knn=25.0,
):
    """
    Default: optimize ``jawOpen`` + pytorch3d ``s,R,T`` on inner ``[17:]`` + optional jawline KNN.
    """
    flame_pts, _, _ = flame_landmarks_68(
        flame_model_path,
        flame_lmk_embedding_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )
    flame_verts, _ = load_flame_neutral_vertices(
        flame_model_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )
    n_pairs = landmark_pair_count(flame_pts, landmark_indices, landmark_start)
    print(
        f"  FLAME landmarks: {len(flame_pts)} pts, paired n={n_pairs} "
        f"(landmark_start={landmark_start})"
    )

    if optimize_jaw:
        opt = optimize_ict_jaw_open(
            ict_vertices,
            landmark_indices,
            expression_shape_modes,
            expression_names,
            flame_pts,
            landmark_start=landmark_start,
            jaw_min=jaw_min,
            jaw_max=jaw_max,
            n_jaw_steps=n_jaw_steps,
            initial_jaw=ict_jaw_open,
            use_rigid=True,
            device=device,
            flame_verts=flame_verts,
            w_jaw_knn=w_jaw_knn,
        )
        jaw = opt["flame_similarity_ict_jaw_open"]
        err_mean = opt["flame_similarity_lmk_err_mean"]
        err_max = opt["flame_similarity_lmk_err_max"]
        out = {
            "flame_similarity_landmark_start": int(landmark_start),
            "flame_similarity_n_pairs": int(n_pairs),
            "flame_similarity_lmk_err_mean": float(err_mean),
            "flame_similarity_lmk_err_max": float(err_max),
            "flame_similarity_jaw_knn_mean": float(opt.get("flame_similarity_jaw_knn_mean", 0.0)),
            "flame_similarity_use_processed_faces": bool(use_processed_faces),
            "flame_similarity_use_canonical_pose": bool(use_canonical_pose),
            "flame_similarity_ict_jaw_open": float(jaw),
            "flame_similarity_s": opt["flame_similarity_s"],
            "flame_similarity_T": opt["flame_similarity_T"],
        }
        if "flame_alignment_s" in opt:
            out["flame_alignment_s"] = opt["flame_alignment_s"]
            out["flame_alignment_R"] = opt["flame_alignment_R"]
            out["flame_alignment_T"] = opt["flame_alignment_T"]
            out["flame_alignment_n_pairs"] = int(n_pairs)
            out["flame_alignment_lmk_err_mean"] = float(err_mean)
            out["flame_alignment_lmk_err_max"] = float(err_max)
        return out

    jaw = float(ict_jaw_open)
    _, err_max, s, R, T, _, err_mean, jaw_knn = _landmark_rigid_error(
        ict_vertices,
        landmark_indices,
        expression_shape_modes,
        expression_names,
        flame_pts,
        jaw_open=jaw,
        landmark_start=landmark_start,
        device=device,
        flame_verts=flame_verts,
        w_jaw_knn=w_jaw_knn,
    )
    return {
        "flame_similarity_s": np.float32(1.0),
        "flame_similarity_T": np.zeros(3, dtype=np.float32),
        "flame_alignment_s": np.float32(s),
        "flame_alignment_R": np.asarray(R, dtype=np.float32),
        "flame_alignment_T": np.asarray(T, dtype=np.float32),
        "flame_similarity_landmark_start": int(landmark_start),
        "flame_similarity_n_pairs": int(n_pairs),
        "flame_similarity_lmk_err_mean": float(err_mean),
        "flame_similarity_lmk_err_max": float(err_max),
        "flame_similarity_jaw_knn_mean": float(jaw_knn),
        "flame_alignment_n_pairs": int(n_pairs),
        "flame_alignment_lmk_err_mean": float(err_mean),
        "flame_alignment_lmk_err_max": float(err_max),
        "flame_similarity_use_processed_faces": bool(use_processed_faces),
        "flame_similarity_use_canonical_pose": bool(use_canonical_pose),
        "flame_similarity_ict_jaw_open": float(jaw),
    }


def fit_rigid_alignment_fields(
    model_dict,
    *,
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    landmark_indices=None,
    landmark_start=LANDMARK_START_FLAME_PAIRING,
    use_processed_faces=False,
    use_canonical_pose=True,
    device="cpu",
):
    """
    Single pytorch3d ``s,R,T``: ``neutral + jawOpen`` ICT landmarks → FLAME ``[17:]``.
    Same as ``optimize_ict_expression_to_flame`` (not coarse-then-rigid compose).
    """
    d = model_dict
    lmk_idx = landmark_indices if landmark_indices is not None else d["landmark_indices"]
    v_jaw = ict_mesh_with_jaw(np.asarray(d["neutral_mesh"], dtype=np.float64), d)
    ict_pts = ict_landmarks_from_vertices(v_jaw, lmk_idx, landmark_start)

    flame_pts, _, _ = flame_landmarks_68(
        flame_model_path,
        flame_lmk_embedding_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )
    flame_p, ict_p = pair_landmarks_for_alignment(flame_pts, ict_pts, flame_landmark_start=landmark_start)

    s, R, T = fit_similarity_rigid(ict_p, flame_p, device=device)
    aligned = apply_flame_alignment(ict_p, s, R, T)
    err = np.linalg.norm(aligned - flame_p, axis=1)

    return {
        "flame_alignment_s": np.float32(s),
        "flame_alignment_R": np.asarray(R, dtype=np.float32),
        "flame_alignment_T": np.asarray(T, dtype=np.float32),
        "flame_alignment_lmk_err_mean": float(err.mean()),
        "flame_alignment_lmk_err_max": float(err.max()),
        "flame_alignment_n_pairs": int(len(flame_p)),
    }


def compute_bake_flame_alignment(
    model_dict,
    *,
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    landmark_indices=None,
    landmark_start=17,
    use_processed_faces=False,
    use_canonical_pose=True,
    device="cpu",
):
    """
  Rigid ``s,R,T`` on landmarks after coarse ``s,T`` + optimized jaw.
  Returns fields to merge into ``ict_facekit_torch.npy`` (composed transform).
    """
    d = model_dict
    lmk_idx = landmark_indices if landmark_indices is not None else d["landmark_indices"]
    v_coarse = ict_mesh_coarse_aligned(d["neutral_mesh"], d)
    ict_pts = ict_landmarks_from_vertices(v_coarse, lmk_idx, landmark_start)

    flame_pts, _, _ = flame_landmarks_68(
        flame_model_path,
        flame_lmk_embedding_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )
    flame_p, ict_p = pair_landmarks_for_alignment(flame_pts, ict_pts, flame_landmark_start=landmark_start)

    s1, R1, t1 = fit_similarity_rigid(ict_p, flame_p, device=device)
    s0, t0 = flame_similarity_from_npy(d)
    s_tot, R_tot, T_tot = compose_st_then_similarity(s0, t0, s1, R1, t1)

    aligned = apply_flame_alignment(ict_p, s_tot, R_tot, T_tot)
    err = np.linalg.norm(aligned - flame_p, axis=1)

    return {
        "flame_alignment_s": np.float32(s_tot),
        "flame_alignment_R": np.asarray(R_tot, dtype=np.float32),
        "flame_alignment_T": np.asarray(T_tot, dtype=np.float32),
        "flame_alignment_rigid_s": np.float32(s1),
        "flame_alignment_lmk_err_mean": float(err.mean()),
        "flame_alignment_lmk_err_max": float(err.max()),
        "flame_alignment_n_pairs": int(len(flame_p)),
    }


def merge_flame_alignment_into_npy(npy_path, alignment_fields):
    """Update ``ict_facekit_torch.npy`` with bake ``flame_alignment_*`` keys."""
    path = Path(npy_path)
    d = np.load(path, allow_pickle=True).item()
    d.update(alignment_fields)
    np.save(str(path), d)
    return d


def compute_ict_flame_alignment_for_npy(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    *,
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    use_processed_faces=False,
    use_canonical_pose=True,
    landmark_start=LANDMARK_START_FLAME_PAIRING,
    ict_jaw_open=0.75,
    optimize_jaw=True,
    jaw_min=0.0,
    jaw_max=1.2,
    n_jaw_steps=25,
    coarse_st_only=False,
    device="cpu",
    w_jaw_knn=25.0,
):
    """
    Npy-build alignment matching ``optimize_ict_expression_to_flame``.

    FLAME: zero exp + canonical pose; ICT: neutral + ``jawOpen``; pytorch3d ``s,R,T`` on ``[17:]``.
    Stored in ``flame_alignment_*`` (``flame_similarity_s/T`` = identity).

    ``coarse_st_only=True``: jaw grid + uniform ``s,T`` only (no rotation).
    """
    if coarse_st_only:
        return compute_ict_flame_similarity_coarse_st(
            ict_vertices,
            landmark_indices,
            expression_shape_modes,
            expression_names,
            flame_model_path=flame_model_path,
            flame_lmk_embedding_path=flame_lmk_embedding_path,
            use_processed_faces=use_processed_faces,
            use_canonical_pose=use_canonical_pose,
            landmark_start=landmark_start,
            ict_jaw_open=ict_jaw_open,
            optimize_jaw=optimize_jaw,
            jaw_min=jaw_min,
            jaw_max=jaw_max,
            n_jaw_steps=n_jaw_steps,
            device=device,
        )
    return compute_ict_flame_similarity(
        ict_vertices,
        landmark_indices,
        expression_shape_modes,
        expression_names,
        flame_model_path=flame_model_path,
        flame_lmk_embedding_path=flame_lmk_embedding_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        landmark_start=landmark_start,
        ict_jaw_open=ict_jaw_open,
        optimize_jaw=optimize_jaw,
        jaw_min=jaw_min,
        jaw_max=jaw_max,
        n_jaw_steps=n_jaw_steps,
        device=device,
        w_jaw_knn=w_jaw_knn,
    )


def compute_ict_flame_similarity_coarse_st(
    ict_vertices,
    landmark_indices,
    expression_shape_modes,
    expression_names,
    *,
    flame_model_path=None,
    flame_lmk_embedding_path=None,
    use_processed_faces=False,
    use_canonical_pose=True,
    landmark_start=17,
    ict_jaw_open=0.75,
    optimize_jaw=True,
    jaw_min=0.0,
    jaw_max=1.2,
    n_jaw_steps=25,
    device="cpu",
):
    """Legacy coarse ``s,T`` only (no ``flame_alignment_*``)."""
    flame_pts, _, _ = flame_landmarks_68(
        flame_model_path,
        flame_lmk_embedding_path,
        use_processed_faces=use_processed_faces,
        use_canonical_pose=use_canonical_pose,
        device=device,
    )
    n_pairs = landmark_pair_count(flame_pts, landmark_indices, landmark_start)

    if optimize_jaw:
        opt = optimize_ict_jaw_open(
            ict_vertices,
            landmark_indices,
            expression_shape_modes,
            expression_names,
            flame_pts,
            landmark_start=landmark_start,
            jaw_min=jaw_min,
            jaw_max=jaw_max,
            n_jaw_steps=n_jaw_steps,
            initial_jaw=ict_jaw_open,
            use_rigid=False,
            device=device,
        )
        s = opt["flame_similarity_s"]
        t = opt["flame_similarity_T"]
        jaw = opt["flame_similarity_ict_jaw_open"]
        err_mean = opt["flame_similarity_lmk_err_mean"]
        err_max = opt["flame_similarity_lmk_err_max"]
    else:
        jaw = float(ict_jaw_open)
        err_mean, err_max, s, t, _ = _landmark_st_error(
            ict_vertices,
            landmark_indices,
            expression_shape_modes,
            expression_names,
            flame_pts,
            jaw_open=jaw,
            landmark_start=landmark_start,
        )

    return {
        "flame_similarity_s": np.float32(s),
        "flame_similarity_T": np.asarray(t, dtype=np.float32),
        "flame_similarity_landmark_start": int(landmark_start),
        "flame_similarity_n_pairs": int(n_pairs),
        "flame_similarity_lmk_err_mean": float(err_mean),
        "flame_similarity_lmk_err_max": float(err_max),
        "flame_similarity_use_processed_faces": bool(use_processed_faces),
        "flame_similarity_use_canonical_pose": bool(use_canonical_pose),
        "flame_similarity_ict_jaw_open": float(jaw),
    }
