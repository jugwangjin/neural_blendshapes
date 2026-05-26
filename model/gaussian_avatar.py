"""
ICT mesh-embedded Gaussians: surface (face_idx+bary) incl. sparse sclera + eye occlusion.

Full-head npy: 12 ``usemtl`` texture maps (``material_names``), 17 geometry charts.
Surface layout samples skin/sockets + ``M_Sclera*`` + ``M_EyeOcclusion``; lacrimal /
iris / lashes / eye-blend / teeth tris are excluded from Gaussians but remain in npy.
Gaussian ``uv`` uses ``triangle_uv_local`` (per-map 0–1) when present.
"""

import torch
import torch.nn as nn

from utils.barycentric import sample_normals, sample_surface
from utils.sampling import build_surface_gaussian_layout


def anchor_mask_from_triangles(face_idx, faces, anchor_vertex_ids):
    """True if any corner of the Gaussian's triangle is in ``anchor_vertex_ids``."""
    n_verts = int(faces.max().item()) + 1
    v_mask = torch.zeros(n_verts, dtype=torch.bool, device=faces.device)
    v_mask[anchor_vertex_ids.long()] = True
    tri = faces[face_idx.long()]
    return v_mask[tri].any(dim=-1)


class GaussianAvatar(nn.Module):
    """Surface Gaussians on ICT triangles (skin, sockets, sclera, eye occlusion)."""

    def __init__(
        self,
        ict,
        face_idx,
        bary,
        *,
        uv=None,
        is_gum=None,
        is_h_pin=None,
        deformer=None,
        sh_dim=3,
        n_semantic_classes=7,
    ):
        super().__init__()
        self.ict = ict
        self.deformer = deformer
        self.sh_dim = sh_dim
        self.n_semantic_classes = n_semantic_classes
        n = face_idx.shape[0]
        self.register_buffer("face_idx", face_idx.long())
        self.register_buffer("bary", bary.float())
        if uv is not None and uv.numel() > 0:
            self.register_buffer("uv", uv.float())
        else:
            self.register_buffer("uv", torch.zeros(0, 2))

        if is_gum is None:
            is_gum = torch.zeros(n, dtype=torch.bool)
        if is_h_pin is None:
            is_h_pin = torch.zeros(n, dtype=torch.bool)
        self.register_buffer("is_gum", is_gum.bool())
        self.register_buffer("is_h_pin", is_h_pin.bool())

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
        opacity_init = float(torch.logit(torch.tensor(0.75)))
        self.opacity = nn.Parameter(torch.full((n, 1), opacity_init))
        self.color = nn.Parameter(torch.zeros(n, sh_dim))

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
        k_face=12,
        k_head=8,
        k_mouth_socket=1,
        k_mouth_interior=1,
        k_eye_socket=1,
        k_eyeball_sclera=4,
        k_eye_occlusion=4,
        k_per_face=None,
        sh_dim=3,
        n_semantic_classes=7,
        gum_h_sigma_scale=4.0,
        gaussian_scale_knn_k=4,
        gaussian_scale_knn_factor=1.0,
        **_,
    ):
        device = ict.neutral_mesh.device
        if k_per_face is not None:
            k_face = k_head = k_per_face
            k_mouth_socket = k_mouth_interior = k_eye_socket = max(1, k_per_face // 4)
            k_eyeball_sclera = k_eye_occlusion = max(1, k_per_face // 2)

        face_idx, bary, _, uv, is_gum, is_h_pin = build_surface_gaussian_layout(
            ict,
            ict.faces,
            k_face=k_face,
            k_head=k_head,
            k_mouth_socket=k_mouth_socket,
            k_mouth_interior=k_mouth_interior,
            k_eye_socket=k_eye_socket,
            k_eyeball_sclera=k_eyeball_sclera,
            k_eye_occlusion=k_eye_occlusion,
            device=device,
        )

        model = cls(
            ict,
            face_idx,
            bary,
            uv=uv,
            is_gum=is_gum,
            is_h_pin=is_h_pin,
            deformer=deformer,
            sh_dim=sh_dim,
            n_semantic_classes=n_semantic_classes,
        )
        h_scale = torch.ones(face_idx.shape[0], dtype=torch.float32, device=device)
        h_scale[is_gum] = float(gum_h_sigma_scale)
        h_scale[is_h_pin] = 0.0
        model.register_buffer("h_sigma_scale", h_scale)

        model._init_surface_semantics()
        model._init_knn_scales(
            k=gaussian_scale_knn_k,
            scale_factor=gaussian_scale_knn_factor,
        )
        if ict.has_texture_maps():
            tid = ict.face_texture_map_id[model.face_idx].long()
            model.register_buffer("face_texture_map_id", tid)
        return model

    def _init_knn_scales(self, k=3, scale_factor=1.0):
        from utils.gaussian_scale_init import init_module_log_scale, surface_gaussian_xyz

        xyz = surface_gaussian_xyz(
            self.ict,
            self.face_idx,
            self.bary,
            h=self.h,
            h_sigma_scale=getattr(self, "h_sigma_scale", None),
        )
        init_module_log_scale(self, xyz, k=k, scale_factor=scale_factor)

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
        h_eff = self.h
        if getattr(self, "h_sigma_scale", None) is not None:
            h_eff = self.h * self.h_sigma_scale.unsqueeze(-1)
        xyz = xyz_base + h_eff * vn
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
            "is_h_pin": self.is_h_pin,
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
        use_pose_scale=False,
        pose_weight_fixed=None,
        rotate_about_centroid=False,
        pose_zero_tz=False,
    ):
        verts_posed = None
        deformed = None
        if tracker_out is not None:
            if self.deformer is None:
                raise ValueError("GaussianAvatar.deformer is required when forward(tracker_out=...)")
            c_eff = tracker_out["coeffs"] if apply_expression_deform else None
            pose_scale = tracker_out.get("pose_scale") if use_pose_scale else None
            deformed = self.deformer(
                mp_coeffs_corr=tracker_out["coeffs"],
                expression_weights=tracker_out.get("ict_expression_weights"),
                pose_rotation_6d=tracker_out["pose_residual"],
                pose_translation=tracker_out["translation_residual"],
                pose_scale=pose_scale,
                c_eff=c_eff,
                expr_delta=expr_delta,
                apply_expression_deform=apply_expression_deform,
                pose_weight_fixed=pose_weight_fixed,
                rotate_about_centroid=rotate_about_centroid,
                pose_zero_tz=pose_zero_tz,
            )
            verts_posed = deformed["verts_posed"]
            verts = verts_posed[0]

        if verts is None:
            raise ValueError("forward requires tracker_out or verts")
        faces = self.ict.faces if faces is None else faces

        surface_out = self._forward_surface(verts, faces)

        if self.anchor_vertex_ids.numel() > 0:
            is_anchor = anchor_mask_from_triangles(
                surface_out["face_idx"], faces, self.anchor_vertex_ids
            )
        else:
            is_anchor = torch.ones(
                surface_out["xyz"].shape[0], dtype=torch.bool, device=verts.device
            )

        out = {
            "surface": surface_out,
            "face": surface_out,
            "xyz": surface_out["xyz"],
            "scale": surface_out["scale"],
            "rotation": surface_out["rotation"],
            "opacity": surface_out["opacity"],
            "color": surface_out["color"],
            "h": surface_out["h"],
            "is_anchor_surface": is_anchor,
            "is_eyeball_surface": surface_out["is_h_pin"],
            "iris_control_xyz": None,
        }
        if verts_posed is not None:
            out["verts_posed"] = verts_posed
            out["mesh_xyz"] = verts_posed
            if deformed is not None and "expr_delta" in deformed:
                out["expr_delta"] = deformed["expr_delta"]
        if surface_out.get("sem_prob") is not None:
            out["sem_prob"] = surface_out["sem_prob"]
        return out
