"""
Vectorized Triangle Walking (mesh adjacency traversal) for Barycentric coordinates.
Enforces partition of unity and bounds constraints, allowing Gaussians to walk across faces.
"""

import torch
from utils.ict_texture_maps import bary_to_texture_chart_uv
from utils.ict_regions import surface_triangle_code_table
from utils.mesh_ops import barycentric_3d


def bary_from_uv(u, v):
    return torch.stack([1.0 - u - v, u, v], dim=-1)


def bary_uv_out_of_bounds(bary_uv):
    u = bary_uv[:, 0]
    v = bary_uv[:, 1]
    return (u < 0.0) | (v < 0.0) | ((u + v) > 1.0)


def clamp_normalize_bary(bary, eps=1e-5):
    bary = torch.clamp(bary, min=eps)
    return bary / bary.sum(dim=-1, keepdim=True)


class TriangleWalker:
    """
    Precomputed mesh topology for post-step barycentric coordinate walking.

    The learnable state lives on the Gaussian surface (`face_idx`, `bary_uv`).
    This object only keeps immutable mesh data needed to re-express escaped
    barycentric coordinates on adjacent faces.
    """

    def __init__(self, faces, vertices, *, adj_faces=None, max_iterations=3):
        if vertices.ndim == 3:
            vertices = vertices[0]
        self.faces = faces.long()
        self.vertices = vertices
        self.adj_faces = adj_faces if adj_faces is not None else build_face_adjacency(self.faces)
        self.max_iterations = max_iterations

    @classmethod
    def from_ict(cls, ict, *, device=None, max_iterations=3):
        faces = ict.faces
        vertices = ict.canonical[0]
        if device is not None:
            faces = faces.to(device)
            vertices = vertices.to(device)
        return cls(faces, vertices, max_iterations=max_iterations)

    def step(self, surface, optimizer=None):
        return walk_barycentric_surface(
            surface,
            self.faces,
            self.vertices,
            self.adj_faces,
            optimizer=optimizer,
            max_iterations=self.max_iterations,
        )


def build_face_adjacency(faces, num_vertices=None):
    """
    Constructs a vectorized face adjacency table.
    
    faces: [F, 3] tensor of vertex indices
    Returns adj_faces: [F, 3] tensor where adj_faces[f, i] is the face index sharing edge i.
    Edge i is defined as:
      - Edge 0: vertices (1, 2) of face f (opposite to vertex 0)
      - Edge 1: vertices (2, 0) of face f (opposite to vertex 1)
      - Edge 2: vertices (0, 1) of face f (opposite to vertex 2)
    """
    F = faces.shape[0]
    if num_vertices is None:
        num_vertices = int(faces.max().item()) + 1
    
    # Define edges: edge0 (1-2), edge1 (2-0), edge2 (0-1)
    e0_v1, e0_v2 = faces[:, 1], faces[:, 2]
    e1_v1, e1_v2 = faces[:, 2], faces[:, 0]
    e2_v1, e2_v2 = faces[:, 0], faces[:, 1]
    
    # Helper to construct a canonical edge ID
    def edge_id(v1, v2):
        min_v = torch.minimum(v1, v2)
        max_v = torch.maximum(v1, v2)
        return min_v.long() * num_vertices + max_v.long()
    
    e0_id = edge_id(e0_v1, e0_v2)
    e1_id = edge_id(e1_v1, e1_v2)
    e2_id = edge_id(e2_v1, e2_v2)
    
    # Pack all edges: [3*F]
    all_edge_ids = torch.cat([e0_id, e1_id, e2_id])
    all_face_ids = torch.arange(F, device=faces.device).repeat(3)
    all_edge_indices = torch.cat([
        torch.zeros(F, dtype=torch.long, device=faces.device),
        torch.ones(F, dtype=torch.long, device=faces.device),
        torch.full((F,), 2, dtype=torch.long, device=faces.device)
    ])
    
    # Sort by edge ID to group adjacent faces together
    sorted_ids, perm = torch.sort(all_edge_ids)
    sorted_faces = all_face_ids[perm]
    sorted_edge_indices = all_edge_indices[perm]
    
    # Find identical adjacent edge IDs
    mask = sorted_ids[:-1] == sorted_ids[1:]
    
    # Adjacency table initialized to -1 (meaning boundary)
    adj_faces = torch.full((F, 3), -1, dtype=torch.long, device=faces.device)
    
    # For adjacent pairs (i, i+1), they share an edge
    idx_left = torch.where(mask)[0]
    idx_right = idx_left + 1
    
    f_left = sorted_faces[idx_left]
    e_left = sorted_edge_indices[idx_left]
    
    f_right = sorted_faces[idx_right]
    e_right = sorted_edge_indices[idx_right]
    
    # Left face adjacent to right face along left edge, and vice-versa
    adj_faces[f_left, e_left] = f_right
    adj_faces[f_right, e_right] = f_left
    
    return adj_faces


@torch.no_grad()
def walk_barycentric_surface(surface, faces, vertices, adj_faces, *, optimizer=None, max_iterations=3):
    """
    Move out-of-face barycentric coordinates to adjacent faces.

    This is vectorized over only the out-of-bounds subset. The first check is a
    cheap full-array bounds test; the heavier face/vertex gathers run only when
    at least one Gaussian crossed a triangle edge.
    """
    if not hasattr(surface, "bary_uv"):
        return 0

    out_of_bounds = bary_uv_out_of_bounds(surface.bary_uv)

    if not out_of_bounds.any():
        return 0

    device = surface.face_idx.device
    if faces.device != device:
        faces = faces.to(device=device)
    if vertices.device != device or vertices.dtype != surface.bary_uv.dtype:
        vertices = vertices.to(device=device, dtype=surface.bary_uv.dtype)
    if adj_faces.device != device:
        adj_faces = adj_faces.to(device=device)

    face_idx = surface.face_idx.clone()
    bary_uv = surface.bary_uv.data.clone()
    shifted_tensors = []
    active_idx = torch.where(out_of_bounds)[0]

    for _ in range(max_iterations):
        if active_idx.numel() == 0:
            break

        u = bary_uv[active_idx, 0]
        v = bary_uv[active_idx, 1]
        bary_active = bary_from_uv(u, v)
        iter_oob = (bary_active < 0.0).any(dim=-1)
        if not iter_oob.any():
            break

        oob_local = torch.where(iter_oob)[0]
        oob_idx = active_idx[oob_local]
        bary_oob = bary_active[oob_local]
        most_neg_vertex = bary_oob.argmin(dim=-1)
        curr_face = face_idx[oob_idx]
        next_face = adj_faces[curr_face, most_neg_vertex]

        boundary_mask = next_face == -1
        if boundary_mask.any():
            b_idx = oob_idx[boundary_mask]
            b_local = oob_local[boundary_mask]
            bary_b = clamp_normalize_bary(bary_active[b_local])
            bary_uv[b_idx] = bary_b[:, 1:3]
            shifted_tensors.append(b_idx)

        walk_mask = ~boundary_mask
        if walk_mask.any():
            w_oob_idx = oob_idx[walk_mask]
            w_curr_face = curr_face[walk_mask]
            w_next_face = next_face[walk_mask]

            curr_tri = vertices[faces[w_curr_face]]
            next_tri = vertices[faces[w_next_face]]
            w_local = oob_local[walk_mask]
            bary_w = bary_active[w_local]

            # Preserve the current 3D point as much as possible by re-projecting it
            # into the neighboring triangle coordinate frame.
            point = (curr_tri * bary_w.unsqueeze(-1)).sum(dim=1)
            new_bary = barycentric_3d(point, next_tri[:, 0], next_tri[:, 1], next_tri[:, 2])

            face_idx[w_oob_idx] = w_next_face
            bary_uv[w_oob_idx] = new_bary[:, 1:3]
            shifted_tensors.append(w_oob_idx)
            active_idx = w_oob_idx
        else:
            active_idx = active_idx.new_empty(0)

    if active_idx.numel() > 0:
        u = bary_uv[active_idx, 0]
        v = bary_uv[active_idx, 1]
        bary_active = bary_from_uv(u, v)
        unresolved = (bary_active < 0.0).any(dim=-1)
    else:
        unresolved = torch.zeros(0, dtype=torch.bool, device=device)

    if unresolved.any():
        unresolved_idx = active_idx[torch.where(unresolved)[0]]
        bary_unresolved = clamp_normalize_bary(bary_active[unresolved])
        bary_uv[unresolved_idx] = bary_unresolved[:, 1:3]
        shifted_tensors.append(unresolved_idx)

    surface.face_idx.copy_(face_idx)
    surface.bary_uv.data.copy_(bary_uv)

    if len(shifted_tensors) > 0:
        shifted_tensor = torch.cat(shifted_tensors).unique()
        _update_shifted_surface_metadata(surface, face_idx, bary_uv, shifted_tensor)

        if optimizer is not None:
            reset_optimizer_moments(optimizer, surface.bary_uv, shifted_tensor)
        return int(shifted_tensor.numel())
    return 0


def _update_shifted_surface_metadata(surface, face_idx, bary_uv, shifted_tensor):
    shifted_u = bary_uv[shifted_tensor, 0]
    shifted_v = bary_uv[shifted_tensor, 1]
    shifted_bary = bary_from_uv(shifted_u, shifted_v)

    if surface.uv.numel() > 0:
        surface.uv[shifted_tensor] = bary_to_texture_chart_uv(
            face_idx[shifted_tensor], shifted_bary, surface.ict
        )
    if hasattr(surface, "face_texture_map_id") and surface.face_texture_map_id is not None:
        surface.face_texture_map_id[shifted_tensor] = surface.ict.face_texture_map_id[
            face_idx[shifted_tensor]
        ].long()
    if hasattr(surface, "face_region_code") and surface.face_region_code is not None:
        code_by_face = surface_triangle_code_table(
            surface.ict.faces, surface.ict, surface.face_region_code.device
        )
        surface.face_region_code[shifted_tensor] = code_by_face[face_idx[shifted_tensor]]


def reset_optimizer_moments(optimizer, param, index):
    state = optimizer.state.get(param, None)
    if state is None:
        return
    if "exp_avg" in state:
        state["exp_avg"][index] = 0.0
    if "exp_avg_sq" in state:
        state["exp_avg_sq"][index] = 0.0


def perform_triangle_walking(avatar, adj_faces, optimizer, max_iterations=3):
    """
    Backward-compatible wrapper. Prefer `TriangleWalker.step(surface, optimizer)`.
    """
    surface = avatar.surface
    faces = surface.ict.faces
    vertices = surface.ict.canonical[0]
    return walk_barycentric_surface(
        surface,
        faces,
        vertices,
        adj_faces,
        optimizer=optimizer,
        max_iterations=max_iterations,
    )
