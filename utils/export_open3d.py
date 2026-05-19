"""Open3D colored point cloud export (Gaussians / mesh vertices)."""

from pathlib import Path

import numpy as np
import torch


def logit_colors_to_rgb(color_logit):
    c = color_logit.detach().float()
    if c.shape[-1] > 3:
        c = c[..., :3]
    return torch.sigmoid(c).clamp(0.0, 1.0)


def _subsample(n, max_points, seed):
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def save_colored_point_cloud(path, xyz, rgb, max_points=100000, seed=0):
    """
    Write ``.ply`` / ``.pcd`` via Open3D.

    ``xyz`` [N,3], ``rgb`` [N,3] in [0, 1] or logits (auto-sigmoid if values outside [0,1]).
    """
    import open3d as o3d

    xyz_np = xyz.detach().float().cpu().numpy().reshape(-1, 3)
    rgb_t = rgb.detach().float() if torch.is_tensor(rgb) else torch.tensor(rgb, dtype=torch.float32)
    if rgb_t.max() > 1.01 or rgb_t.min() < -0.01:
        rgb_t = logit_colors_to_rgb(rgb_t)
    if rgb_t.shape[-1] > 3:
        rgb_t = rgb_t[..., :3]
    rgb_np = rgb_t.clamp(0.0, 1.0).cpu().numpy().reshape(-1, 3)

    idx = _subsample(xyz_np.shape[0], max_points, seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np[idx])
    pcd.colors = o3d.utility.Vector3dVector(rgb_np[idx])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    return int(idx.shape[0])


def vertex_region_codes(ict, device):
    """
    Per-vertex region code (same palette as sanity layout).

    0 mouth_interior, 1 mouth_socket, 2 eye_socket, 3 head_neck, 4 face, -1 eyeball/teeth.
    """
    n_verts = int(ict.neutral_mesh.shape[1])
    code = torch.full((n_verts,), 4, dtype=torch.long, device=device)

    def mark(cid, ids):
        if ids is None or len(ids) == 0:
            return
        idx = torch.tensor(list(ids), device=device, dtype=torch.long)
        code[idx] = cid

    mark(3, getattr(ict, "not_face_indices", []))
    mark(2, getattr(ict, "eye_socket_left_indices", []))
    mark(2, getattr(ict, "eye_socket_right_indices", []))
    mark(1, getattr(ict, "mouth_socket_indices", []))
    mark(
        0,
        getattr(
            ict,
            "mouth_interior_vertex_indices",
            getattr(ict, "gums_tongue_indices", []),
        ),
    )
    skip = set(getattr(ict, "eyeball_indices", [])) | set(getattr(ict, "teeth_indices", []))
    if skip:
        mark(-1, list(skip))
    return code


def region_palette_rgb(region_rgb_dict, device):
    """``{code: (r,g,b)}`` → tensor [max_code+1, 3]."""
    max_code = max(region_rgb_dict.keys())
    pal = torch.zeros(max_code + 1, 3, device=device)
    for code, rgb in region_rgb_dict.items():
        if code >= 0:
            pal[code] = torch.tensor(rgb, device=device, dtype=torch.float32)
    return pal


def mesh_vertex_colors_from_regions(ict, region_rgb, device):
    codes = vertex_region_codes(ict, device)
    pal = region_palette_rgb(region_rgb, device)
    rgb = pal[codes.clamp(min=0)]
    rgb[codes < 0] = 0.5
    return rgb


def save_gaussian_point_cloud(path, avatar_out, max_points=100000, seed=0):
    return save_colored_point_cloud(
        path, avatar_out["xyz"], avatar_out["color"], max_points=max_points, seed=seed
    )


def save_mesh_point_cloud(path, verts, rgb, max_points=100000, seed=0):
    if torch.is_tensor(verts) and verts.dim() == 3:
        verts = verts[0]
    if torch.is_tensor(rgb) and rgb.dim() == 3:
        rgb = rgb[0]
    return save_colored_point_cloud(path, verts, rgb, max_points=max_points, seed=seed)
