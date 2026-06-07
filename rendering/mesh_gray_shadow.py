"""Minimal nvdiffrast mesh render: flat gray + directional Lambert shading."""

from __future__ import annotations

import nvdiffrast.torch as dr
import torch

from losses.mesh_silhouette import (
    _exclude_faces_by_vertex_ids,
    _gl_projection_from_camera,
    _gl_view_from_camera,
    _raster_ctx,
)
from utils.mesh_ops import vertex_normals


def _mvp_from_camera(camera, *, width: int, height: int, near: float, far: float, device, dtype):
    proj = _gl_projection_from_camera(
        camera, width=width, height=height, near=near, far=far, device=device, dtype=dtype
    )
    view = _gl_view_from_camera(camera, device=device, dtype=dtype)
    return proj @ view


def _camera_space_normals(mesh_xyz, faces, camera):
    """World normals → camera space, unit length. Returns [B,V,3]."""
    b = mesh_xyz.shape[0]
    device = mesh_xyz.device
    dtype = mesh_xyz.dtype
    rows = []
    for i in range(b):
        n = vertex_normals(mesh_xyz[i], faces)
        rows.append(n)
    vn_world = torch.stack(rows, dim=0)
    R = camera.R.to(device=device, dtype=dtype)
    if R.ndim == 2:
        vn_cam = torch.einsum("ij,bvj->bvi", R, vn_world)
    else:
        vn_cam = torch.einsum("bij,bvj->bvi", R, vn_world)
    return vn_cam / vn_cam.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _default_tracking_light_dirs(device, dtype):
    """
    Camera-space key + fill (OpenCV: +Z forward, Y down).

    Key mimics FLARE ``draw_mediapipe.py`` preview: camera-side illumination with
    soft env-like fill from upper-right. Front-facing normals ≈ -Z, so key uses -Z.
    """
    key = torch.tensor([0.12, -0.35, -1.0], device=device, dtype=dtype)
    fill = torch.tensor([-0.55, -0.15, 0.75], device=device, dtype=dtype)
    return key / key.norm(), fill / fill.norm()


def _half_lambert(ndotl, wrap: float = 0.35):
    return ((ndotl + wrap) / (1.0 + wrap)).clamp(0.0, 1.0)


@torch.no_grad()
def render_mesh_gray_shadow(
    mesh_xyz,
    faces,
    camera,
    *,
    image_size: int | None = None,
    near: float = 0.01,
    far: float = 100.0,
    base_gray: float = 0.72,
    ambient: float = 0.38,
    light_dir=None,
    fill_weight: float = 0.28,
    exclude_vertex_ids=None,
    background: float = 1.0,
):
    """
    Rasterize posed ICT mesh with simple gray Lambert shading (FLARE-style preview).

    Args:
        mesh_xyz: [B,V,3] world-space vertices.
        faces: [F,3] int triangle indices (shared topology).
        camera: training ``Camera`` (OpenCV world → image).
        exclude_vertex_ids: drop any triangle touching these vertex indices (e.g. eyelashes).
        light_dir: optional camera-space direction; default front key + soft fill.

    Returns:
        rgb float [B,H,W,3] in [0, 1] on white background.
    """
    b, v, _ = mesh_xyz.shape
    device = mesh_xyz.device
    dtype = mesh_xyz.dtype
    w = int(getattr(camera, "width", image_size or 512))
    h = int(getattr(camera, "height", image_size or 512))

    mesh_xyz = mesh_xyz.contiguous()
    mvp = _mvp_from_camera(camera, width=w, height=h, near=near, far=far, device=device, dtype=dtype)
    ones = torch.ones((b, v, 1), device=device, dtype=dtype)
    posw = torch.cat([mesh_xyz, ones], dim=-1)
    clip = torch.bmm(posw, mvp.t().unsqueeze(0).expand(b, -1, -1)).contiguous()

    faces_eff = _exclude_faces_by_vertex_ids(faces, exclude_vertex_ids)
    tri = faces_eff.to(device=device, dtype=torch.int32).contiguous()
    if tri.numel() == 0:
        bg = torch.full((b, h, w, 3), float(background), device=device, dtype=dtype)
        return bg

    ctx = _raster_ctx(device)
    rast, _ = dr.rasterize(ctx, clip, tri, resolution=(h, w))
    rast = rast.contiguous()

    vn_cam = _camera_space_normals(mesh_xyz, faces_eff, camera).contiguous()
    n_attr, _ = dr.interpolate(vn_cam, rast, tri)
    n_attr = n_attr / n_attr.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    if light_dir is None:
        ld_key, ld_fill = _default_tracking_light_dirs(device, dtype)
    else:
        ld_key = torch.tensor(light_dir, device=device, dtype=dtype)
        ld_key = ld_key / ld_key.norm()
        ld_fill = ld_key

    ndotl_key = _half_lambert(
        (n_attr * ld_key.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True)
    )
    if light_dir is None and float(fill_weight) > 0.0:
        ndotl_fill = (n_attr * ld_fill.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True).clamp(min=0.0)
        ndotl = (1.0 - float(fill_weight)) * ndotl_key + float(fill_weight) * ndotl_fill
    else:
        ndotl = ndotl_key
    shade = float(ambient) + (1.0 - float(ambient)) * ndotl
    gray = float(base_gray) * shade
    rgb = gray.expand(-1, -1, -1, 3)

    alpha, _ = dr.interpolate(ones.contiguous(), rast, tri)
    alpha = dr.antialias(alpha, rast, clip, tri)
    bg = torch.full((3,), float(background), device=device, dtype=dtype)
    rgb = rgb * alpha + bg.view(1, 1, 1, 3) * (1.0 - alpha)
    rgb = dr.antialias(rgb, rast, clip, tri)
    return rgb.clamp(0.0, 1.0)
