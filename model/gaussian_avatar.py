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


def _normalize_quat_wxyz(q):
    n = torch.sqrt(torch.sum(q**2, dim=-1, keepdim=True) + 1e-12)
    return q / n


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _triangle_frames(verts, faces, fallback_frames=None):
    """Per-face orthonormal frames [e1, e2, n] in world coordinates."""
    tri = verts[faces.long()]
    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]
    e1 = v1 - v0
    e1_len = torch.sqrt(torch.sum(e1**2, dim=-1, keepdim=True) + 1e-12)
    n = torch.cross(e1, v2 - v0, dim=-1)
    n_len = torch.sqrt(torch.sum(n**2, dim=-1, keepdim=True) + 1e-12)
    degenerate = (e1_len.squeeze(-1) < 1e-8) | (n_len.squeeze(-1) < 1e-8)

    e1 = e1 / e1_len.clamp(min=1e-8)
    n = n / n_len.clamp(min=1e-8)
    e2 = torch.cross(n, e1, dim=-1)
    e2_len = torch.sqrt(torch.sum(e2**2, dim=-1, keepdim=True) + 1e-12)
    e2 = e2 / e2_len.clamp(min=1e-8)
    frames = torch.stack([e1, e2, n], dim=-1)

    if degenerate.any() and fallback_frames is not None:
        fb = fallback_frames[degenerate]
        frames = frames.clone()
        frames[degenerate] = fb
    return frames


def _rotmat_to_quat_wxyz(R):
    """Rotation matrix [N,3,3] -> quaternion [N,4] (w,x,y,z)."""
    def _sign_nonzero(x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    m00 = R[:, 0, 0]
    m11 = R[:, 1, 1]
    m22 = R[:, 2, 2]
    m01 = R[:, 0, 1]
    m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]
    m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]
    m21 = R[:, 2, 1]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-12))
    qx = 0.5 * _sign_nonzero(m21 - m12) * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12))
    qy = 0.5 * _sign_nonzero(m02 - m20) * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12))
    qz = 0.5 * _sign_nonzero(m10 - m01) * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12))
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return _normalize_quat_wxyz(q)


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
        max_scale=0.008,
    ):
        super().__init__()
        self.ict = ict
        self.deformer = deformer
        self.sh_dim = sh_dim
        self.n_semantic_classes = n_semantic_classes
        self.max_scale = max_scale
        n = face_idx.shape[0]
        self.register_buffer("face_idx", face_idx.long())
        self.bary_uv = nn.Parameter(bary[:, 1:3].float().clone())
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
        # opacity is sigmoid(logit) so logit(0.5)=0.0 => start from medium opacity
        opacity_init = float(torch.logit(torch.tensor(0.9)))
        self.opacity = nn.Parameter(torch.full((n, 1), opacity_init))
        # View-independent base color plus pose/expression-conditioned deltas.
        self.color = nn.Parameter(torch.zeros(n, 3))
        # [N, 3(out RGB), 3(in pose angles)].
        self.color_pose = nn.Parameter(torch.zeros(n, 3, 3))
        # [N, K(=53 coeffs), 3(out RGB)].
        self.color_expression = nn.Parameter(
            torch.zeros(n, int(getattr(ict, "num_expression", 53)), 3)
        )

        self.register_buffer(
            "anchor_vertex_ids",
            torch.tensor(list(ict.face_indices), dtype=torch.long),
        )
        template_verts = ict.neutral_mesh
        if template_verts.ndim == 3:
            template_verts = template_verts[0]
        self.register_buffer(
            "template_tri_frames",
            _triangle_frames(template_verts, ict.faces),
        )

    @property
    def bary(self):
        u = self.bary_uv[:, 0:1]
        v = self.bary_uv[:, 1:2]
        w = 1.0 - u - v
        return torch.cat([w, u, v], dim=-1)

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
        face_center_init=False,
        max_scale=0.008,
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
            face_center_init=face_center_init,
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
            max_scale=max_scale,
        )
        h_scale = torch.ones(face_idx.shape[0], dtype=torch.float32, device=device)
        h_scale[is_gum] = float(gum_h_sigma_scale)
        h_scale[is_h_pin] = 0.0
        model.register_buffer("h_sigma_scale", h_scale)

        model._init_surface_semantics()
        model._init_surface_region_codes()
        model._init_knn_scales(
            k=gaussian_scale_knn_k,
            scale_factor=gaussian_scale_knn_factor,
        )
        if ict.has_texture_maps():
            tid = ict.face_texture_map_id[model.face_idx].long()
            model.register_buffer("face_texture_map_id", tid)
        return model

    @classmethod
    def from_checkpoint_state(
        cls,
        ict,
        deformer,
        state_dict,
        *,
        max_scale=0.008,
    ):
        """
        Rebuild surface layout from a training checkpoint ``avatar`` state_dict
        (after densification), not from ``from_ict`` sampling counts.
        """
        device = ict.neutral_mesh.device
        face_idx = state_dict["face_idx"].to(device=device, dtype=torch.long)
        bary_uv = state_dict["bary_uv"].to(device=device, dtype=torch.float32)
        u = bary_uv[:, 0:1]
        v = bary_uv[:, 1:2]
        w = (1.0 - u - v).clamp(min=0.0)
        bary_init = torch.cat([w, u, v], dim=-1)

        uv = state_dict.get("uv")
        if uv is None or uv.numel() == 0:
            uv = torch.zeros(0, 2, device=device, dtype=torch.float32)
        else:
            uv = uv.to(device=device, dtype=torch.float32)

        n_sem = 0
        if "sem_logits" in state_dict and state_dict["sem_logits"] is not None:
            n_sem = int(state_dict["sem_logits"].shape[1])

        sh_dim = int(state_dict["color"].shape[-1])
        model = cls(
            ict,
            face_idx,
            bary_init,
            uv=uv,
            is_gum=state_dict["is_gum"].to(device=device),
            is_h_pin=state_dict["is_h_pin"].to(device=device),
            deformer=deformer,
            sh_dim=sh_dim,
            n_semantic_classes=n_sem,
            max_scale=max_scale,
        )
        load_state = dict(state_dict)
        for k in ("color_pose", "color_expression"):
            if k in load_state and tuple(load_state[k].shape) != tuple(getattr(model, k).shape):
                load_state.pop(k)
        model.load_state_dict(load_state, strict=False)
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

    def _init_surface_region_codes(self):
        from utils.ict_regions import classify_surface_triangles_batch

        codes = classify_surface_triangles_batch(
            self.face_idx,
            self.ict.faces,
            self.ict,
            self.face_idx.device,
        )
        self.register_buffer("face_region_code", codes)

    @staticmethod
    def _vertex_normals(verts, faces):
        from utils.mesh_ops import vertex_normals

        return vertex_normals(verts, faces)

    def _pose_angle_vector(self, pose_rotation_6d):
        """
        Pose rotation 6D -> frontal-reference signed Euler angles (rad), [N,3].
        Excludes translation/scale effects by construction.
        """
        if pose_rotation_6d is None:
            return torch.zeros(self.face_idx.shape[0], 3, device=self.face_idx.device)
        from utils.mesh_ops import rotation_6d_to_matrix

        R = rotation_6d_to_matrix(pose_rotation_6d.float())  # [B,3,3]
        # XYZ convention (roll-x, pitch-y, yaw-z) from rotation matrix.
        sy = torch.sqrt((R[:, 0, 0] * R[:, 0, 0]) + (R[:, 1, 0] * R[:, 1, 0])).clamp(min=1e-8)
        singular = sy < 1e-6

        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        pitch = torch.atan2(-R[:, 2, 0], sy)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        roll_s = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
        pitch_s = torch.atan2(-R[:, 2, 0], sy)
        yaw_s = torch.zeros_like(yaw)

        roll = torch.where(singular, roll_s, roll)
        pitch = torch.where(singular, pitch_s, pitch)
        yaw = torch.where(singular, yaw_s, yaw)

        ang_vec = torch.stack([roll, pitch, yaw], dim=-1).mean(dim=0, keepdim=True)
        ang_vec = ang_vec.to(device=self.face_idx.device, dtype=self.color.dtype)
        return ang_vec.expand(self.face_idx.shape[0], 3)

    def _forward_surface(
        self,
        verts,
        faces,
        expr_coeff=None,
        pose_angle_vec=None,
        enable_color_pose=True,
        enable_color_expression=True,
    ):
        if verts.ndim == 3:
            verts = verts[0]
        bary = self.bary
        xyz_base = sample_surface(verts, faces, self.face_idx, bary)
        vn = sample_normals(
            self._vertex_normals(verts, faces), faces, self.face_idx, bary
        )
        h_eff = self.h
        if getattr(self, "h_sigma_scale", None) is not None:
            h_eff = self.h * self.h_sigma_scale.unsqueeze(-1)
        xyz = xyz_base + h_eff * vn
        scale = torch.exp(self.log_scale)
        opacity = torch.sigmoid(self.opacity)
        tri_frames_tmpl = self.template_tri_frames[self.face_idx]
        tri_frames_curr = _triangle_frames(
            verts, faces, fallback_frames=self.template_tri_frames
        )[self.face_idx]
        tri_rot = tri_frames_curr @ tri_frames_tmpl.transpose(1, 2)
        mesh_quat = _rotmat_to_quat_wxyz(tri_rot)
        local_quat = _normalize_quat_wxyz(self.rotation)
        rotation = _normalize_quat_wxyz(_quat_mul_wxyz(mesh_quat, local_quat))
        color = self.color
        if enable_color_pose:
            if pose_angle_vec is None:
                pose_angle_vec = torch.zeros(
                    self.face_idx.shape[0], 3, device=self.face_idx.device, dtype=self.color.dtype
                )
            color = color + torch.einsum("nij,nj->ni", self.color_pose, pose_angle_vec)
        if enable_color_expression:
            if expr_coeff is None:
                expr_coeff = torch.zeros(
                    self.color_expression.shape[1],
                    device=self.face_idx.device,
                    dtype=self.color.dtype,
                )
            else:
                if expr_coeff.ndim == 2:
                    expr_coeff = expr_coeff.mean(dim=0)
                expr_coeff = expr_coeff.to(
                    device=self.face_idx.device,
                    dtype=self.color.dtype,
                )
            color = color + torch.einsum("nkr,k->nr", self.color_expression, expr_coeff)

        out = {
            "xyz": xyz,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity,
            "color": color,
            "h": self.h,
            "face_idx": self.face_idx,
            "bary": bary,
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
        skip_surface=False,
        enable_color_pose=True,
        enable_color_expression=True,
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
                return_unposed=True,
            )
            verts_posed = deformed["verts_posed"]
            verts = verts_posed[0]

        if verts is None:
            raise ValueError("forward requires tracker_out or verts")
        faces = self.ict.faces if faces is None else faces

        if skip_surface:
            out = {
                "mesh_xyz": verts_posed,
                "verts_posed": verts_posed,
                "xyz": verts,
            }
            if deformed is not None and "expr_delta" in deformed:
                out["expr_delta"] = deformed["expr_delta"]
            return out

        expr_coeff = None
        pose_angle_vec = None
        if deformed is not None:
            expr_coeff = tracker_out.get("coeffs") if tracker_out is not None else None
        if tracker_out is not None:
            pose_angle_vec = self._pose_angle_vector(tracker_out.get("pose_residual"))

        surface_out = self._forward_surface(
            verts,
            faces,
            expr_coeff=expr_coeff,
            pose_angle_vec=pose_angle_vec,
            enable_color_pose=enable_color_pose,
            enable_color_expression=enable_color_expression,
        )

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
