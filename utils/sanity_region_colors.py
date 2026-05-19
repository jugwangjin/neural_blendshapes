"""Sanity-render region colors (flat + world-Y gradients on face / head-neck / eyes)."""

import torch

from utils.barycentric import sample_surface
from utils.export_open3d import vertex_region_codes
from utils.ict_regions import classify_surface_triangles_batch

# Flat regions
MOUTH_INTERIOR_RGB = (1.0, 0.05, 0.05)
MOUTH_SOCKET_RGB = (0.55, 0.05, 0.15)
EYE_SOCKET_RGB = (1.0, 1.0, 1.0)

# Face: higher Y → brighter peach, lower Y → darker peach
PEACH_RGB_HI = (1.0, 0.85, 0.72)
PEACH_RGB_LO = (0.68, 0.48, 0.38)

# Head / neck: low Y (neck) → white, high Y (crown) → black
HEAD_RGB_CROWN = (0.0, 0.0, 0.0)
HEAD_RGB_NECK = (1.0, 1.0, 1.0)

# Eye Gaussians: iris control UV bbox → pupil black, sclera white
IRIS_PUPIL_RGB = (0.0, 0.0, 0.0)
EYE_SCLERA_RGB = (1.0, 1.0, 1.0)
IRIS_UV_RECT_MARGIN = 0.015

FLAT_REGION_RGB = {
    0: MOUTH_INTERIOR_RGB,
    1: MOUTH_SOCKET_RGB,
    2: EYE_SOCKET_RGB,
}


def rgb_to_logit(rgb):
    if torch.is_tensor(rgb):
        t = rgb.float().clamp(1e-4, 1.0 - 1e-4)
    else:
        t = torch.tensor(rgb, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(t / (1.0 - t))


def _y_gradient_rgb(y, sel, rgb_lo, rgb_hi):
    """``rgb_lo`` at min Y, ``rgb_hi`` at max Y among selected points."""
    y_sel = y[sel]
    y_lo = y_sel.min()
    y_hi = y_sel.max()
    t = ((y_sel - y_lo) / (y_hi - y_lo + 1e-8)).clamp(0.0, 1.0)
    lo = torch.tensor(rgb_lo, device=y.device, dtype=torch.float32)
    hi = torch.tensor(rgb_hi, device=y.device, dtype=torch.float32)
    return lo + t.unsqueeze(1) * (hi - lo)


def iris_uv_rectangle(eyes, device, margin=None):
    """
    Axis-aligned rect in sclera *local* chart around ``iris_control_uv`` (MP iris pentagon).
    """
    margin = IRIS_UV_RECT_MARGIN if margin is None else float(margin)
    ctrl = eyes.iris_control_uv.to(device=device, dtype=torch.float32)
    umin = ctrl[:, 0].min() - margin
    umax = ctrl[:, 0].max() + margin
    vmin = ctrl[:, 1].min() - margin
    vmax = ctrl[:, 1].max() + margin
    return umin, umax, vmin, vmax


def mask_in_iris_uv_rect(uv, eyes, device, margin=None):
    umin, umax, vmin, vmax = iris_uv_rectangle(eyes, device, margin=margin)
    return (
        (uv[:, 0] >= umin)
        & (uv[:, 0] <= umax)
        & (uv[:, 1] >= vmin)
        & (uv[:, 1] <= vmax)
    )


def eye_gaussian_rgb(eyes, side, device, gaze_uv=None):
    """
    Per-eye Gaussian RGB [n_per_eye, 3]: iris rect (black), sclera (white).
    """
    uv = eyes.uv.detach().float().to(device)
    if gaze_uv is not None:
        g = torch.as_tensor(gaze_uv, device=device, dtype=uv.dtype).reshape(1, 2)
        uv = uv + g
    if side == "R" and eyes.mirror_right_u:
        uv = torch.stack([1.0 - uv[:, 0], uv[:, 1]], dim=-1)

    n = uv.shape[0]
    colors = torch.full((n, 3), EYE_SCLERA_RGB[0], device=device, dtype=torch.float32)
    colors[:, 1] = EYE_SCLERA_RGB[1]
    colors[:, 2] = EYE_SCLERA_RGB[2]
    pupil = mask_in_iris_uv_rect(uv, eyes, device)
    if pupil.any():
        colors[pupil] = torch.tensor(IRIS_PUPIL_RGB, device=device, dtype=torch.float32)
    return colors


def eye_gaussian_rgb_shared(eyes, device, gaze_left=None, gaze_right=None):
    """
    RGB for shared ``EyeTextureGaussians.color`` [n_per_eye, 3] (chart-local UV).

    Pupil mask in canonical chart; forward duplicates color for L/R instantiate.
    """
    uv = eyes.uv.detach().float().to(device)
    if gaze_left is not None:
        g = torch.as_tensor(gaze_left, device=device, dtype=uv.dtype).reshape(1, 2)
        uv = uv + g
    is_pupil = mask_in_iris_uv_rect(uv, eyes, device)

    colors = torch.full((uv.shape[0], 3), EYE_SCLERA_RGB[0], device=device, dtype=torch.float32)
    colors[:, 1] = EYE_SCLERA_RGB[1]
    colors[:, 2] = EYE_SCLERA_RGB[2]
    colors[is_pupil] = torch.tensor(IRIS_PUPIL_RGB, device=device, dtype=torch.float32)
    return colors


def surface_gaussian_rgb(avatar, ict, verts, device):
    """Per surface Gaussian RGB [N, 3] in [0, 1]."""
    codes = classify_surface_triangles_batch(avatar.face_idx, ict.faces, ict, device)
    v = verts.detach().float().reshape(-1, 3).to(device)
    xyz = sample_surface(v, ict.faces.to(device), avatar.face_idx, avatar.bary)
    y = xyz[:, 1]
    n = codes.shape[0]
    colors = torch.zeros(n, 3, device=device, dtype=torch.float32)

    for code, rgb in FLAT_REGION_RGB.items():
        sel = codes == code
        if sel.any():
            colors[sel] = torch.tensor(rgb, device=device, dtype=torch.float32)

    sel_face = codes == 4
    if sel_face.any():
        colors[sel_face] = _y_gradient_rgb(y, sel_face, PEACH_RGB_LO, PEACH_RGB_HI)

    sel_head = codes == 3
    if sel_head.any():
        colors[sel_head] = _y_gradient_rgb(y, sel_head, HEAD_RGB_NECK, HEAD_RGB_CROWN)

    return colors


def mesh_vertex_rgb(ict, verts, device):
    """Per-vertex RGB [V, 3] for Open3D mesh PLY."""
    codes = vertex_region_codes(ict, device)
    v = verts.detach().float().reshape(-1, 3).to(device)
    y = v[:, 1]
    n_verts = v.shape[0]
    rgb = torch.zeros(n_verts, 3, device=device, dtype=torch.float32)

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
