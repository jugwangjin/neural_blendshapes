"""
ICT mesh-embedded Gaussians: surface (face_idx+bary) + sclera eye UV + optional accessory.

Holds ``ict`` (and optional ``deformer``) so ``tracker_out`` maps directly to posed mesh + gaze.
"""

import torch
import torch.nn as nn

from model.accessory_gaussians import AccessoryGaussians
from model.eye_texture_gaussians import EyeTextureGaussians
from utils.barycentric import sample_normals, sample_surface
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
    """
    Single avatar module: surface Gaussians on ICT triangles + eye texture Gaussians.

    ``surface`` property aliases ``self`` for training helpers that expect ``avatar.surface.h``.
    """

    def __init__(
        self,
        ict,
        face_idx,
        bary,
        *,
        uv=None,
        eyes: EyeTextureGaussians = None,
        accessory: AccessoryGaussians = None,
        deformer=None,
        sh_dim=3,
        n_semantic_classes=7,
    ):
        super().__init__()
        self.ict = ict
        self.deformer = deformer
        self.eyes = eyes
        self.accessory = accessory
        self.sh_dim = sh_dim
        self.n_semantic_classes = n_semantic_classes

        n = face_idx.shape[0]
        self.register_buffer("face_idx", face_idx.long())
        self.register_buffer("bary", bary.float())
        if uv is not None and uv.numel() > 0:
            self.register_buffer("uv", uv.float())
        else:
            self.register_buffer("uv", torch.zeros(0, 2))

        self.sem_logits = None
        self.register_buffer("sem_prob_fixed", None)
        self.register_buffer("sem_anchor", None)
        self.register_buffer("sem_frozen_dims", None)
        if n_semantic_classes > 0:
            self.sem_logits = nn.Parameter(torch.zeros(n, n_semantic_classes))

        self.h = nn.Parameter(torch.zeros(n, 1))
        self.log_scale = nn.Parameter(torch.zeros(n, 3))
        self.rotation = nn.Parameter(torch.zeros(n, 4))
        self.rotation.data[:, 0] = 1.0
        self.opacity = nn.Parameter(torch.zeros(n, 1))
        self.color = nn.Parameter(torch.zeros(n, sh_dim))

        self.texture_meshes = None
        self.register_buffer(
            "anchor_vertex_ids",
            torch.tensor(list(ict.face_indices), dtype=torch.long),
        )

    @property
    def surface(self):
        return self

    @property
    def n_gaussians(self):
        return self.face_idx.shape[0]

    @classmethod
    def from_ict(
        cls,
        ict,
        deformer=None,
        k_face=8,
        k_head=8,
        k_mouth_socket=1,
        k_mouth_interior=2,
        k_eye_socket=1,
        k_per_face=None,
        n_eye_per_side=1024,
        n_accessory_gaussians=0,
        sh_dim=3,
        gaze_uv_range=0.12,
        learn_gaze_refine=True,
        n_semantic_classes=7,
        gum_h_sigma_scale=4.0,
        accessory_anchor_xyz=None,
        accessory_tangent_basis=None,
    ):
        device = ict.neutral_mesh.device
        if k_per_face is not None:
            k_face = k_head = k_per_face
            k_mouth_socket = k_mouth_interior = k_eye_socket = max(1, k_per_face // 4)

        face_idx, bary, _, uv, is_gum = build_surface_gaussian_layout(
            ict,
            ict.faces,
            k_face=k_face,
            k_head=k_head,
            k_mouth_socket=k_mouth_socket,
            k_mouth_interior=k_mouth_interior,
            k_eye_socket=k_eye_socket,
            device=device,
        )

        mirror_right_u = bool(getattr(ict, "eye_uv_mirror_right_u", False))
        eyes = EyeTextureGaussians(
            n_per_eye=n_eye_per_side,
            sh_dim=sh_dim,
            gaze_uv_range=gaze_uv_range,
            learn_gaze_refine=learn_gaze_refine,
            n_semantic_classes=n_semantic_classes,
            mirror_right_u=mirror_right_u,
            ict=ict,
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

        model = cls(
            ict,
            face_idx,
            bary,
            uv=uv,
            eyes=eyes,
            accessory=accessory,
            deformer=deformer,
            sh_dim=sh_dim,
            n_semantic_classes=n_semantic_classes,
        )
        h_scale = torch.ones(face_idx.shape[0], dtype=torch.float32, device=device)
        h_scale[is_gum] = float(gum_h_sigma_scale)
        model.register_buffer("h_sigma_scale", h_scale)
        model.register_buffer("is_gum", is_gum)

        model.texture_meshes = TextureSpaceMeshes.from_ict(ict)
        model._init_surface_semantics()
        return model

    def _init_surface_semantics(self):
        from rendering.gaussian_semantics import init_face_gaussian_semantics

        if self.n_semantic_classes == 0:
            return
        init_face_gaussian_semantics(
            self,
            self.face_idx,
            self.bary,
            self.ict,
            self.ict.faces,
        )

    @staticmethod
    def _vertex_normals(verts, faces):
        from utils.mesh_ops import vertex_normals

        return vertex_normals(verts, faces)

    def _forward_surface(self, verts, faces):
        if verts.ndim == 3:
            verts = verts[0]
        xyz_base = sample_surface(verts, faces, self.face_idx, self.bary)
        vn = sample_normals(
            self._vertex_normals(verts, faces), faces, self.face_idx, self.bary
        )
        xyz = xyz_base + self.h * vn
        scale = torch.exp(self.log_scale).clamp(max=0.05)
        opacity = torch.sigmoid(self.opacity)
        out = {
            "xyz": xyz,
            "scale": scale,
            "rotation": self.rotation,
            "opacity": opacity,
            "color": self.color,
            "h": self.h,
            "face_idx": self.face_idx,
            "bary": self.bary,
            "normals": vn,
            "group": "surface",
        }
        if self.sem_prob_fixed is not None:
            out["sem_prob"] = self.sem_prob_fixed
        elif self.sem_logits is not None:
            from rendering.gaussian_semantics import semantic_probs_with_anchor

            out["sem_prob"] = semantic_probs_with_anchor(
                self.sem_logits, self.sem_anchor, self.sem_frozen_dims
            )
        return out

    def forward(
        self,
        tracker_out=None,
        verts=None,
        faces=None,
        expr_delta=None,
        apply_expression_deform=False,
        gaze_uv_left=None,
        gaze_uv_right=None,
    ):
        """
        Primary path: ``avatar(tracker_out=corr, apply_expression_deform=...)`` with ``deformer`` set.

        Legacy path: ``avatar(verts, faces, gaze_uv_left=..., gaze_uv_right=...)``.
        """
        verts_posed = None
        deformed = None
        if tracker_out is not None:
            if self.deformer is None:
                raise ValueError("GaussianAvatar.deformer is required when forward(tracker_out=...)")
            c_eff = tracker_out["coeffs"] if apply_expression_deform else None
            deformed = self.deformer(
                mp_coeffs_corr=tracker_out["coeffs"],
                pose_rotation_6d=tracker_out["pose_residual"],
                pose_translation=tracker_out["translation_residual"],
                pose_scale=tracker_out.get("pose_scale"),
                c_eff=c_eff,
                expr_delta=expr_delta,
                apply_expression_deform=apply_expression_deform,
            )
            verts_posed = deformed["verts_posed"]
            verts = verts_posed[0]
            gaze_uv_left = tracker_out["gaze_uv_left"]
            gaze_uv_right = tracker_out["gaze_uv_right"]

        if verts is None:
            raise ValueError("forward requires tracker_out or verts")
        faces = self.ict.faces if faces is None else faces

        device = verts.device
        dtype = torch.float32
        if gaze_uv_left is not None:
            gaze_uv_left = torch.as_tensor(gaze_uv_left, device=device, dtype=dtype)
        if gaze_uv_right is not None:
            gaze_uv_right = torch.as_tensor(gaze_uv_right, device=device, dtype=dtype)

        surface_out = self._forward_surface(verts, faces)

        if self.anchor_vertex_ids.numel() > 0:
            surface_out["is_anchor_surface"] = anchor_mask_from_triangles(
                surface_out["face_idx"], faces, self.anchor_vertex_ids
            )
        else:
            surface_out["is_anchor_surface"] = torch.ones(
                surface_out["xyz"].shape[0], dtype=torch.bool, device=device
            )

        tm = self.texture_meshes
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
            "gaze_uv_left": eye_out["gaze_offset_left"],
            "gaze_uv_right": eye_out["gaze_offset_right"],
        }
        if verts_posed is not None:
            out["verts_posed"] = verts_posed
            out["mesh_xyz"] = verts_posed
            if deformed is not None and "expr_delta" in deformed:
                out["expr_delta"] = deformed["expr_delta"]

        if surface_out.get("sem_prob") is not None:
            sem_parts = [surface_out["sem_prob"], eye_out["sem_prob"]]
            if self.accessory is not None and self.accessory.n_gaussians > 0:
                sem_parts.append(acc_out["sem_prob"])
            out["sem_prob"] = torch.cat(sem_parts, dim=0)

        if self.accessory is not None and self.accessory.n_gaussians > 0:
            out["accessory"] = acc_out

        return out
