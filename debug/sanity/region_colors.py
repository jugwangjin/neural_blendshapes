"""Debug-render region colors (flat + world-Y gradients on face / head-neck / eyes)."""

from pathlib import Path

import numpy as np
import torch

from utils.barycentric import sample_surface, vertices2landmarks
from utils.ict_regions import classify_surface_triangles_batch, eye_occlusion_layout_face_indices
from utils.mediapipe_blendshapes import IRIS_MP, LEFT_IRIS_MP, RIGHT_IRIS_MP

from .export_open3d import vertex_region_codes

MOUTH_INTERIOR_RGB = (1.0, 0.05, 0.05)
MOUTH_SOCKET_RGB = (0.55, 0.05, 0.15)
EYE_SOCKET_RGB = (1.0, 1.0, 1.0)

PEACH_RGB_HI = (1.0, 0.85, 0.72)
PEACH_RGB_LO = (0.68, 0.48, 0.38)

HEAD_RGB_CROWN = (0.0, 0.0, 0.0)
HEAD_RGB_NECK = (1.0, 1.0, 1.0)

EYE_SCLERA_GAUSSIAN_RGB = (1.0, 1.0, 1.0)
EYE_OCCLUSION_GAUSSIAN_RGB = (0.92, 0.92, 0.95)
EYE_OCCLUSION_IRIS_RGB = (0.0, 0.0, 0.0)
LANDMARK_FACE_RGB = (0.0, 1.0, 0.0)

FLAT_REGION_RGB = {
    0: MOUTH_INTERIOR_RGB,
    1: MOUTH_SOCKET_RGB,
    2: EYE_SOCKET_RGB,
    5: EYE_SCLERA_GAUSSIAN_RGB,
    6: EYE_OCCLUSION_GAUSSIAN_RGB,
}


def rgb_to_logit(rgb):
    if torch.is_tensor(rgb):
        t = rgb.float().clamp(1e-4, 1.0 - 1e-4)
    else:
        t = torch.tensor(rgb, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(t / (1.0 - t))


def _y_gradient_rgb(y, sel, rgb_lo, rgb_hi):
    y_sel = y[sel]
    y_lo = y_sel.min()
    y_hi = y_sel.max()
    t = ((y_sel - y_lo) / (y_hi - y_lo + 1e-8)).clamp(0.0, 1.0)
    lo = torch.tensor(rgb_lo, device=y.device, dtype=torch.float32)
    hi = torch.tensor(rgb_hi, device=y.device, dtype=torch.float32)
    return lo + t.unsqueeze(1) * (hi - lo)


def _load_mp_embedding(path):
    from losses.mediapipe_landmark_478 import load_mediapipe_ict_embedding

    return load_mediapipe_ict_embedding(Path(path))


def landmark_face_indices_from_embedding(embedding_path) -> torch.Tensor:
    """Unique ICT triangle indices referenced by MP barycentric embedding."""
    emb = _load_mp_embedding(embedding_path)
    fi = np.unique(np.asarray(emb["ict_lmk_face_idx"], dtype=np.int64))
    return torch.tensor(fi, dtype=torch.long)


def iris_face_indices_on_eye_occlusion(ict, embedding_path) -> torch.Tensor:
    """ICT face indices where MP iris 468–477 hit ``M_EyeOcclusion`` (ray_occ bake)."""
    occ = set(eye_occlusion_layout_face_indices(ict).tolist())
    if not occ:
        return torch.tensor([], dtype=torch.long)
    emb = _load_mp_embedding(embedding_path)
    mp = np.asarray(emb["mp_landmark_indices"], dtype=np.int64)
    fi = np.asarray(emb["ict_lmk_face_idx"], dtype=np.int64)
    faces = sorted({int(fi[i]) for i in range(len(mp)) if int(mp[i]) in IRIS_MP and int(fi[i]) in occ})
    return torch.tensor(faces, dtype=torch.long)


def _iris_embedding_rows(emb, mp_ids):
    mp = np.asarray(emb["mp_landmark_indices"], dtype=np.int64)
    want = set(mp_ids)
    rows = [i for i in range(len(mp)) if int(mp[i]) in want]
    if not rows:
        return None, None
    fi = torch.tensor([int(emb["ict_lmk_face_idx"][i]) for i in rows], dtype=torch.long)
    bary = torch.tensor(
        np.stack([np.asarray(emb["ict_lmk_b_coords"][i], dtype=np.float32) for i in rows]),
        dtype=torch.float32,
    )
    return fi, bary


def iris_landmark_xyz_by_side(verts, faces, embedding_path, device):
    """Left/right iris MP 468–477 positions on deformed mesh [5,3] each (may be 0 rows)."""
    emb = _load_mp_embedding(embedding_path)
    v = verts.detach().float().reshape(1, -1, 3).to(device)
    f = faces.to(device)
    out = {}
    for side, mp_ids in (("L", LEFT_IRIS_MP), ("R", RIGHT_IRIS_MP)):
        fi, bary = _iris_embedding_rows(emb, mp_ids)
        if fi is None:
            out[side] = torch.zeros(0, 3, device=device)
        else:
            out[side] = vertices2landmarks(v, f, fi.to(device), bary.to(device))[0]
    return out


def occlusion_iris_gaussian_mask(
    occ_xyz,
    ict,
    verts,
    faces,
    occ_face_idx,
    *,
    mp_embedding_path,
    device,
    iris_radius_scale=1.35,
):
    """
    True for eye-occlusion Gaussians in the iris landmark neighborhood.

    1) Face match: iris MP baked on ``M_EyeOcclusion`` triangles.
    2) Else 3D: within ``iris_radius_scale`` × per-eye iris pentagon radius.
    """
    n = occ_xyz.shape[0]
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    if n == 0 or mp_embedding_path is None:
        return mask

    iris_faces = iris_face_indices_on_eye_occlusion(ict, mp_embedding_path).to(device)
    if iris_faces.numel() > 0:
        mask = torch.isin(occ_face_idx, iris_faces)
        if mask.any():
            return mask

    by_side = iris_landmark_xyz_by_side(verts, faces, mp_embedding_path, device)
    for iris_xyz in by_side.values():
        iris_pts = iris_xyz.reshape(-1, 3)
        if iris_pts.shape[0] < 2:
            continue
        center = iris_pts.mean(dim=0)
        spread = (iris_pts - center).norm(dim=-1).max().clamp(min=1e-6)
        radius = spread * float(iris_radius_scale)
        dist = torch.cdist(occ_xyz, iris_pts).min(dim=1).values
        mask = mask | (dist <= radius)
    return mask


def surface_gaussian_rgb(avatar, ict, verts, device, *, mp_embedding_path=None, iris_radius_scale=1.35):
    faces = ict.faces.to(device)
    codes = classify_surface_triangles_batch(avatar.face_idx, faces, ict, device)
    v = verts.detach().float().reshape(-1, 3).to(device)
    xyz = sample_surface(v, faces, avatar.face_idx, avatar.bary)
    y = xyz[:, 1]
    colors = torch.zeros(codes.shape[0], 3, device=device, dtype=torch.float32)

    for code, rgb in FLAT_REGION_RGB.items():
        sel = codes == code
        if sel.any():
            colors[sel] = torch.tensor(rgb, device=device, dtype=torch.float32)

    sel_occ = codes == 6
    if sel_occ.any() and mp_embedding_path is not None:
        occ_xyz = xyz[sel_occ]
        occ_fi = avatar.face_idx[sel_occ]
        iris_gauss = occlusion_iris_gaussian_mask(
            occ_xyz,
            ict,
            verts,
            faces,
            occ_fi,
            mp_embedding_path=mp_embedding_path,
            device=device,
            iris_radius_scale=iris_radius_scale,
        )
        if iris_gauss.any():
            colors[sel_occ][iris_gauss] = torch.tensor(
                EYE_OCCLUSION_IRIS_RGB, device=device, dtype=torch.float32
            )

    sel_face = codes == 4
    if sel_face.any():
        colors[sel_face] = _y_gradient_rgb(y, sel_face, PEACH_RGB_LO, PEACH_RGB_HI)

    sel_head = codes == 3
    if sel_head.any():
        colors[sel_head] = _y_gradient_rgb(y, sel_head, HEAD_RGB_NECK, HEAD_RGB_CROWN)

    if mp_embedding_path is not None:
        lmk_faces = landmark_face_indices_from_embedding(mp_embedding_path).to(device)
        if lmk_faces.numel() > 0:
            on_lmk = torch.isin(avatar.face_idx, lmk_faces)
            if on_lmk.any():
                colors[on_lmk] = torch.tensor(LANDMARK_FACE_RGB, device=device, dtype=torch.float32)

    return colors


def surface_gaussian_iris_stats(avatar, ict, verts, device, *, mp_embedding_path):
    """Counts for sanity logs."""
    faces = ict.faces.to(device)
    codes = classify_surface_triangles_batch(avatar.face_idx, faces, ict, device)
    sel_occ = codes == 6
    n_occ = int(sel_occ.sum().item())
    if n_occ == 0 or mp_embedding_path is None:
        return {"n_occ": n_occ, "n_iris_black": 0, "iris_faces_on_occ": 0}
    xyz = sample_surface(
        verts.detach().float().reshape(-1, 3).to(device),
        faces,
        avatar.face_idx,
        avatar.bary,
    )
    iris_gauss = occlusion_iris_gaussian_mask(
        xyz[sel_occ],
        ict,
        verts,
        faces,
        avatar.face_idx[sel_occ],
        mp_embedding_path=mp_embedding_path,
        device=device,
    )
    lmk_faces = landmark_face_indices_from_embedding(mp_embedding_path)
    on_lmk = torch.isin(avatar.face_idx, lmk_faces.to(device))
    return {
        "n_occ": n_occ,
        "n_iris_black": int(iris_gauss.sum().item()),
        "iris_faces_on_occ": int(iris_face_indices_on_eye_occlusion(ict, mp_embedding_path).numel()),
        "landmark_faces": int(lmk_faces.numel()),
        "gaussians_on_landmark_faces": int(on_lmk.sum().item()),
    }


def mesh_vertex_rgb(ict, verts, device):
    codes = vertex_region_codes(ict, device)
    v = verts.detach().float().reshape(-1, 3).to(device)
    y = v[:, 1]
    rgb = torch.zeros(v.shape[0], 3, device=device, dtype=torch.float32)

    for code, color in FLAT_REGION_RGB.items():
        sel = codes == code
        if sel.any():
            rgb[sel] = torch.tensor(color, device=device, dtype=torch.float32)

    sel_face = codes == 4
    if sel_face.any():
        rgb[sel_face] = _y_gradient_rgb(y, sel_face, PEACH_RGB_LO, PEACH_RGB_HI)

    sel_head = codes == 3
    if sel_head.any():
        rgb[sel_head] = _y_gradient_rgb(y, sel_head, HEAD_RGB_NECK, HEAD_RGB_CROWN)

    rgb[codes < 0] = 0.5
    return rgb
