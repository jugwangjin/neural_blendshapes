"""
UV surface parametrization for mesh-embedded Gaussians (SplattingAvatar-style).

Uses separate verts_uvs / faces_uvs (ICT seam-safe).
"""

from dataclasses import dataclass

import numpy as np
import torch

from utils.barycentric import barycentric_coords_2d


@dataclass
class UVMesh:
    verts: torch.Tensor          # [V, 3]
    faces: torch.Tensor          # [F, 3] full topology
    verts_uvs: torch.Tensor      # [VT, 2]
    faces_uvs: torch.Tensor      # [F, 3] indices into verts_uvs
    active_face_idx: torch.Tensor = None  # subset of faces for this texture space

    @classmethod
    def from_ict_facekit(cls, ict, device="cpu"):
        return cls(
            verts=ict.neutral_mesh[0].to(device),
            faces=ict.faces.to(device),
            verts_uvs=ict.uvs.to(device),
            faces_uvs=ict.uv_faces.to(device),
            active_face_idx=None,
        )


def _triangle_uv_coords(uv_mesh: UVMesh):
    return uv_mesh.verts_uvs[uv_mesh.faces_uvs]


def _search_faces(uv_mesh: UVMesh):
    if uv_mesh.active_face_idx is not None:
        return uv_mesh.active_face_idx.cpu().numpy()
    return np.arange(uv_mesh.faces_uvs.shape[0])


def uv_to_face_bary(uv_points, uv_mesh: UVMesh, eps=1e-4):
    """
    uv_points: [G, 2] in texture space
    returns face_idx [G], bary [G, 3] indexing full face array
    """
    uv_np = uv_points.detach().cpu().numpy()
    tri_uv_all = _triangle_uv_coords(uv_mesh).detach().cpu().numpy()
    search_fi = _search_faces(uv_mesh)

    face_idx = []
    bary = []

    for p in uv_np:
        found = False
        for fi in search_fi:
            tri = tri_uv_all[fi]
            w = barycentric_coords_2d(p, tri[0], tri[1], tri[2])
            if (w >= -eps).all():
                face_idx.append(int(fi))
                bary.append(w.astype(np.float32))
                found = True
                break
        if not found:
            centers = tri_uv_all[search_fi].mean(axis=1)
            local = int(np.argmin(np.linalg.norm(centers - p[None], axis=1)))
            fi = int(search_fi[local])
            tri = tri_uv_all[fi]
            w = barycentric_coords_2d(
                np.clip(p, tri.min(0), tri.max(0)), tri[0], tri[1], tri[2]
            )
            face_idx.append(fi)
            bary.append(w.astype(np.float32))

    face_idx = torch.tensor(face_idx, dtype=torch.long, device=uv_points.device)
    bary = torch.tensor(np.stack(bary), dtype=uv_points.dtype, device=uv_points.device)
    return face_idx, bary


def surface_points_from_uvh(uv, h, uv_mesh: UVMesh):
    """
    uv: [G, 2], h: [G, 1]
    X = S(u,v) + h * N(u,v)   (h=0 for eyeball texture spaces)
    """
    from utils.barycentric import sample_normals, sample_surface
    from utils.mesh_ops import compute_vertex_normals

    face_idx, bary = uv_to_face_bary(uv, uv_mesh)
    v_normals = compute_vertex_normals(uv_mesh.verts, uv_mesh.faces)
    p = sample_surface(uv_mesh.verts, uv_mesh.faces, face_idx, bary)
    n = sample_normals(v_normals, uv_mesh.faces, face_idx, bary)
    xyz = p + h * n
    return xyz, face_idx, bary, n
