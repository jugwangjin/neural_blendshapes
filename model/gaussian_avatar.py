"""Face + per-eye texture-space Gaussians (ICT vertex_parts)."""

import torch
import torch.nn as nn

from model.eye_texture_gaussians import EyeTextureGaussians
from model.uvh_gaussians import UVHGaussians
from utils.texture_spaces import TextureSpaceMeshes, PART_EYEBALL, PART_FACE


def anchor_mask_from_triangles(face_idx, faces, anchor_vertex_ids):
    tri = faces[face_idx]
    anchor = anchor_vertex_ids
    return (tri.unsqueeze(-1) == anchor.view(1, 1, -1)).any(dim=(1, 2))


class GaussianAvatar(nn.Module):
    def __init__(
        self,
        n_face_gaussians,
        n_eye_per_side=64,
        sh_dim=3,
        gaze_uv_range=0.12,
        learn_gaze_refine=True,
    ):
        super().__init__()
        self.face = UVHGaussians(n_face_gaussians, sh_dim=sh_dim)
        self.eyes = EyeTextureGaussians(
            n_per_eye=n_eye_per_side,
            sh_dim=sh_dim,
            gaze_uv_range=gaze_uv_range,
            learn_gaze_refine=learn_gaze_refine,
        )
        self.texture_meshes = None
        self.anchor_vertex_ids = torch.tensor([], dtype=torch.long)

    def set_anchor_vertices(self, face_indices, eyeball_indices):
        """h prior on face skin only; eyeball uses separate texture spaces with h=0."""
        self.anchor_vertex_ids = torch.tensor(list(face_indices), dtype=torch.long)

    @classmethod
    def from_ict(
        cls,
        ict,
        n_face_gaussians,
        n_eye_per_side=64,
        sh_dim=3,
        gaze_uv_range=0.12,
        learn_gaze_refine=True,
    ):
        model = cls(
            n_face_gaussians,
            n_eye_per_side=n_eye_per_side,
            sh_dim=sh_dim,
            gaze_uv_range=gaze_uv_range,
            learn_gaze_refine=learn_gaze_refine,
        )
        model.texture_meshes = TextureSpaceMeshes.from_ict(ict)
        model.set_anchor_vertices(ict.face_indices, ict.eyeball_indices)

        return model

    def forward(self, verts, faces, expression_weights=None, expression_names=None):
        tm = self.texture_meshes
        face_mesh = tm.face
        left_mesh = tm.left_eye
        right_mesh = tm.right_eye

        face_out = self.face(face_mesh, verts=verts, faces=faces)

        if self.anchor_vertex_ids.numel() > 0:
            face_out["is_anchor_surface"] = anchor_mask_from_triangles(
                face_out["face_idx"], faces, self.anchor_vertex_ids
            )
        else:
            face_out["is_anchor_surface"] = torch.ones(
                face_out["xyz"].shape[0], dtype=torch.bool, device=face_out["xyz"].device
            )

        if expression_weights is not None and expression_names is not None:
            self.eyes.gaze_from_expression(expression_weights, expression_names)

        eye_out = self.eyes(left_mesh, right_mesh, verts, faces)

        is_anchor = torch.cat(
            [face_out["is_anchor_surface"], eye_out["is_eyeball_surface"]], dim=0
        )

        return {
            "face": face_out,
            "eyes": eye_out,
            "texture_meshes": tm,
            "xyz": torch.cat([face_out["xyz"], eye_out["xyz"]], dim=0),
            "scale": torch.cat([face_out["scale"], eye_out["scale"]], dim=0),
            "rotation": torch.cat([face_out["rotation"], eye_out["rotation"]], dim=0),
            "opacity": torch.cat([face_out["opacity"], eye_out["opacity"]], dim=0),
            "color": torch.cat([face_out["color"], eye_out["color"]], dim=0),
            "h": torch.cat([face_out["h"], eye_out["h"]], dim=0),
            "is_anchor_surface": is_anchor,
            "is_eyeball_surface": eye_out["is_eyeball_surface"],
            "iris_control_xyz": eye_out["iris_control_xyz"],
            "gaze_uv_left": eye_out["left"]["gaze_offset"],
            "gaze_uv_right": eye_out["right"]["gaze_offset"],
        }
