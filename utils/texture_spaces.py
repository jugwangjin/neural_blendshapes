"""
ICT multi texture-space meshes from vertex_parts.

parts_split (ict_facekit_to_npy.py):
  0: 0-9408       face skin
  1: 9409-11247    mouth socket
  2: 11248-13293   ...
  3: 13294-13677   left eye socket
  4: 13678-14061   right eye socket
  5: 14062-17038   ...
  6: 21451-23020   left eyeball
  7: 23021-24590   right eyeball
"""

import torch

from utils.uv_mesh import UVMesh

# ICT vertex_parts ids per texture atlas / material island
PART_FACE = (0, 1, 2, 5)
PART_LEFT_EYE = (3, 6)
PART_RIGHT_EYE = (4, 7)
PART_EYEBALL = (6, 7)


def _vertex_parts_tensor(vertex_parts, device):
    if torch.is_tensor(vertex_parts):
        return vertex_parts.to(device=device, dtype=torch.long)
    return torch.tensor(vertex_parts, device=device, dtype=torch.long)


def filter_face_indices(faces, vertex_parts, allowed_part_ids):
    """Keep faces whose three corners all belong to allowed_part_ids."""
    allowed = set(allowed_part_ids)
    keep = []
    if torch.is_tensor(vertex_parts):
        vp = vertex_parts
        for fi in range(faces.shape[0]):
            tri = faces[fi]
            if all(int(vp[v]) in allowed for v in tri):
                keep.append(fi)
    else:
        for fi in range(faces.shape[0]):
            tri = faces[fi].tolist()
            if all(vertex_parts[v] in allowed for v in tri):
                keep.append(fi)
    return torch.tensor(keep, dtype=torch.long, device=faces.device)


def build_texture_space_mesh(verts, faces, uvs, uv_faces, vertex_parts, allowed_part_ids, device):
    """
    Submesh UVMesh: same full verts/uvs, restricted face set for one texture space.
    uv_to_face_bary only searches triangles in this space.
    """
    face_idx = filter_face_indices(faces, vertex_parts, allowed_part_ids)
    return UVMesh(
        verts=verts.to(device),
        faces=faces.to(device),
        verts_uvs=uvs.to(device),
        faces_uvs=uv_faces.to(device),
    ), face_idx


class TextureSpaceMeshes:
    """face / left_eye / right_eye UVMeshes + active face index lists."""

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
        vp = ict.vertex_parts

        face_mesh, face_fi = build_texture_space_mesh(
            verts, faces, uvs, uv_faces, vp, PART_FACE, device
        )
        left_mesh, left_fi = build_texture_space_mesh(
            verts, faces, uvs, uv_faces, vp, PART_LEFT_EYE, device
        )
        right_mesh, right_fi = build_texture_space_mesh(
            verts, faces, uvs, uv_faces, vp, PART_RIGHT_EYE, device
        )
        face_mesh.active_face_idx = face_fi
        left_mesh.active_face_idx = left_fi
        right_mesh.active_face_idx = right_fi
        return cls(face_mesh, left_mesh, right_mesh, face_fi, left_fi, right_fi)
