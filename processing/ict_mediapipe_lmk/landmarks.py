import numpy as np
import torch
import trimesh


def sample_bary(vertices, faces, face_idx, bary):
    tri = faces[face_idx]
    pts = vertices[tri]
    return (pts * bary[:, :, None]).sum(axis=1)


def vertices2landmarks(vertices, faces, lmk_face_idx, lmk_bary_coords):
    """
    vertices: [B, V, 3]
    faces: [F, 3]
    lmk_face_idx: [L]
    lmk_bary_coords: [L, 3]
    returns: [B, L, 3]
    """
    tri = faces[lmk_face_idx]
    lmk_vertices = vertices[:, tri]
    return (lmk_vertices * lmk_bary_coords[None, :, :, None]).sum(dim=2)


def barycentric_coords(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def _face_mask_for_vertices(faces, vertex_mask):
    return np.all(vertex_mask[faces], axis=1)


def project_points_to_mesh_bary(points, vertices, faces, face_mask=None):
    if face_mask is not None:
        sub_faces = faces[face_mask]
        global_face_ids = np.where(face_mask)[0]
    else:
        sub_faces = faces
        global_face_ids = np.arange(len(faces), dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=sub_faces, process=False)
    closest_pts, dist, local_face_ids = trimesh.proximity.closest_point(mesh, points)

    ict_face_idx = global_face_ids[local_face_ids]
    bary = []
    for p, face_id in zip(closest_pts, ict_face_idx):
        tri = faces[face_id]
        a, b, c = vertices[tri]
        bary.append(barycentric_coords(p, a, b, c))

    return (
        ict_face_idx.astype(np.int64),
        np.stack(bary, axis=0).astype(np.float32),
        dist.astype(np.float32),
    )
