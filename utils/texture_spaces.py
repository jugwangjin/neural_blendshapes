"""
ICT texture-space meshes from npy region index arrays (not hardcoded part ids).
"""

import torch

from utils.eye_chart import build_sclera_uv_mesh
from utils.ict_regions import filter_triangles_all_vertices_in, surface_allowed_vertices
from utils.uv_mesh import UVMesh


def build_texture_space_mesh(verts, faces, uvs, uv_faces, allowed_vertex_ids, device):
    face_idx = filter_triangles_all_vertices_in(faces, allowed_vertex_ids, device=device)
    return UVMesh(
        verts=verts.to(device),
        faces=faces.to(device),
        verts_uvs=uvs.to(device),
        faces_uvs=uv_faces.to(device),
    ), face_idx


class TextureSpaceMeshes:
    def __init__(self, face, left_eye, right_eye, face_face_idx, left_eye_face_idx, right_eye_face_idx):
        self.face = face
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.face_face_idx = face_face_idx
        self.left_eye_face_idx = left_eye_face_idx
        self.right_eye_face_idx = right_eye_face_idx

    @classmethod
    def from_ict(cls, ict, device=None):
        device = device or ict.neutral_mesh.device
        verts = ict.neutral_mesh[0]
        faces = ict.faces
        uvs = ict.uvs
        uv_faces = ict.uv_faces

        surface_verts = getattr(ict, "surface_sample_vertex_indices", None)
        if surface_verts is None:
            surface_verts = surface_allowed_vertices(ict)

        face_mesh, face_fi = build_texture_space_mesh(
            verts, faces, uvs, uv_faces, surface_verts, device
        )
        left_mesh = build_sclera_uv_mesh(ict, "L", device)
        right_mesh = build_sclera_uv_mesh(ict, "R", device)
        left_fi = left_mesh.active_face_idx
        right_fi = right_mesh.active_face_idx

        face_mesh.active_face_idx = face_fi
        return cls(face_mesh, left_mesh, right_mesh, face_fi, left_fi, right_fi)
