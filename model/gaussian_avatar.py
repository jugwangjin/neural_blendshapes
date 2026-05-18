"""Surface + eye texture + optional accessory Gaussians."""

import torch
import torch.nn as nn

from model.accessory_gaussians import AccessoryGaussians
from model.eye_texture_gaussians import EyeTextureGaussians
from model.surface_gaussians import SurfaceGaussians
from utils.sampling import build_surface_gaussian_layout
from utils.texture_spaces import TextureSpaceMeshes


def anchor_mask_from_triangles(face_idx, faces, anchor_vertex_ids):
    tri = faces[face_idx]
    anchor = anchor_vertex_ids
    return (tri.unsqueeze(-1) == anchor.view(1, 1, -1)).any(dim=(1, 2))


def _random_tangent_basis(n, device):
    a = torch.randn(n, 3, device=device)
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = torch.randn(n, 3, device=device)
    b = b - (b * a).sum(-1, keepdim=True) * a
    b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.stack([a, b], dim=-1)


def init_accessory_anchors(n, neutral_verts, device):
    center = neutral_verts.mean(dim=0)
    extent = neutral_verts.max(dim=0).values - neutral_verts.min(dim=0).values
    spread = extent.max() * 0.12
    anchor = center.unsqueeze(0) + torch.randn(n, 3, device=device) * spread
    basis = _random_tangent_basis(n, device)
    return anchor, basis


class GaussianAvatar(nn.Module):
    def __init__(
        self,
        surface: SurfaceGaussians,
        eyes: EyeTextureGaussians,
        accessory: AccessoryGaussians = None,
        sh_dim=3,
        n_semantic_classes=7,
    ):
        super().__init__()
        self.n_semantic_classes = n_semantic_classes
        self.surface = surface
        self.eyes = eyes
        self.accessory = accessory
        self.texture_meshes = None
        self.anchor_vertex_ids = torch.tensor([], dtype=torch.long)

    def set_anchor_vertices(self, face_indices, eyeball_indices):
        self.anchor_vertex_ids = torch.tensor(list(face_indices), dtype=torch.long)

    @classmethod
    def from_ict(
        cls,
        ict,
        k_per_face=8,
        n_eye_per_side=1024,
        n_accessory_gaussians=0,
        sh_dim=3,
        gaze_uv_range=0.12,
        learn_gaze_refine=True,
        n_semantic_classes=7,
        accessory_anchor_xyz=None,
        accessory_tangent_basis=None,
    ):
        device = ict.neutral_mesh.device
        face_idx, bary, _ = build_surface_gaussian_layout(
            ict.faces, ict.vertex_parts, k_per_face, device=device
        )
        surface = SurfaceGaussians(
            face_idx,
            bary,
            sh_dim=sh_dim,
            n_semantic_classes=n_semantic_classes,
        )
        eyes = EyeTextureGaussians(
            n_per_eye=n_eye_per_side,
            sh_dim=sh_dim,
            gaze_uv_range=gaze_uv_range,
            learn_gaze_refine=learn_gaze_refine,
            n_semantic_classes=n_semantic_classes,
        )
        accessory = None
        if n_accessory_gaussians > 0:
            verts = ict.neutral_mesh[0]
            if accessory_anchor_xyz is None:
                accessory_anchor_xyz, accessory_tangent_basis = init_accessory_anchors(
                    n_accessory_gaussians, verts, device
                )
            accessory = AccessoryGaussians(
                n_accessory_gaussians,
                accessory_anchor_xyz,
                accessory_tangent_basis,
                sh_dim=sh_dim,
                n_semantic_classes=n_semantic_classes,
            )
        model = cls(surface, eyes, accessory, sh_dim=sh_dim, n_semantic_classes=n_semantic_classes)
        model.texture_meshes = TextureSpaceMeshes.from_ict(ict)
        model.set_anchor_vertices(ict.face_indices, ict.eyeball_indices)
        model._init_surface_semantics(ict)
        return model

    def _init_surface_semantics(self, ict):
        from rendering.gaussian_semantics import init_face_gaussian_semantics

        if self.surface.n_semantic_classes == 0:
            return
        init_face_gaussian_semantics(
            self.surface,
            self.surface.face_idx,
            self.surface.bary,
            ict,
            ict.faces,
        )

    def forward(
        self,
        verts,
        faces,
        gaze_uv_left=None,
        gaze_uv_right=None,
    ):
        tm = self.texture_meshes
        surface_out = self.surface(verts, faces)

        if self.anchor_vertex_ids.numel() > 0:
            surface_out["is_anchor_surface"] = anchor_mask_from_triangles(
                surface_out["face_idx"], faces, self.anchor_vertex_ids
            )
        else:
            surface_out["is_anchor_surface"] = torch.ones(
                surface_out["xyz"].shape[0], dtype=torch.bool, device=surface_out["xyz"].device
            )

        eye_out = self.eyes(
            tm.left_eye,
            tm.right_eye,
            verts,
            faces,
            gaze_uv_left=gaze_uv_left,
            gaze_uv_right=gaze_uv_right,
        )

        parts = [surface_out, eye_out]
        if self.accessory is not None and self.accessory.n_gaussians > 0:
            acc_out = self.accessory()
            parts.append(acc_out)

        xyz = torch.cat([p["xyz"] for p in parts], dim=0)
        is_anchor = torch.cat(
            [
                surface_out["is_anchor_surface"],
                eye_out["is_eyeball_surface"],
            ]
            + (
                [torch.zeros(acc_out["xyz"].shape[0], dtype=torch.bool, device=xyz.device)]
                if self.accessory is not None and self.accessory.n_gaussians > 0
                else []
            ),
            dim=0,
        )

        out = {
            "surface": surface_out,
            "face": surface_out,
            "eyes": eye_out,
            "texture_meshes": tm,
            "xyz": xyz,
            "scale": torch.cat([p["scale"] for p in parts], dim=0),
            "rotation": torch.cat([p["rotation"] for p in parts], dim=0),
            "opacity": torch.cat([p["opacity"] for p in parts], dim=0),
            "color": torch.cat([p["color"] for p in parts], dim=0),
            "h": torch.cat([p["h"] for p in parts], dim=0),
            "is_anchor_surface": is_anchor,
            "is_eyeball_surface": torch.cat(
                [
                    torch.zeros(surface_out["xyz"].shape[0], dtype=torch.bool, device=xyz.device),
                    eye_out["is_eyeball_surface"],
                ]
                + (
                    [torch.zeros(acc_out["xyz"].shape[0], dtype=torch.bool, device=xyz.device)]
                    if self.accessory is not None and self.accessory.n_gaussians > 0
                    else []
                ),
                dim=0,
            ),
            "iris_control_xyz": eye_out["iris_control_xyz"],
            "gaze_uv_left": eye_out["left"]["gaze_offset"],
            "gaze_uv_right": eye_out["right"]["gaze_offset"],
        }
        if surface_out.get("sem_prob") is not None:
            sem_parts = [surface_out["sem_prob"], eye_out["sem_prob"]]
            if self.accessory is not None and self.accessory.n_gaussians > 0:
                sem_parts.append(acc_out["sem_prob"])
            out["sem_prob"] = torch.cat(sem_parts, dim=0)
        if self.accessory is not None and self.accessory.n_gaussians > 0:
            out["accessory"] = acc_out
        return out
