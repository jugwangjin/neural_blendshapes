"""
ICT mesh-embedded Gaussians: surface (face_idx+bary) incl. sparse sclera + eye occlusion.

Full-head npy: 12 ``usemtl`` texture maps (``material_names``), 17 geometry charts.
Surface layout samples skin/sockets + ``M_Sclera*`` + ``M_EyeOcclusion``; lacrimal /
iris / lashes / eye-blend charts excluded; teeth tris use sparse Gaussians (semantic mouth_interior).
Gaussian ``uv`` uses ``triangle_uv_local`` (per-map 0–1) when present.
"""

import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_multiply

from model.blendshape_support import precompute_expression_support
from model.mesh_gaussian_pose import MeshGaussianPoseHelper, barycentric_vertex_quaternion
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
        is_teeth=None,
        deformer=None,
        sh_dim=3,
        n_semantic_classes=0,
        max_scale=0.008,
        expression_support_alpha=0.1,
        expression_support_dilate_rings=4,
        expression_support_train_mask=0.25,
        expression_coeff_eps=1e-4,
        color_expression_exclude_mouth_eye=False,
        with_mesh_scaling=True,
        scale_max_clamp_factor=5.0,
    ):
        super().__init__()
        self.ict = ict
        self.deformer = deformer
        self.sh_dim = sh_dim
        self.n_semantic_classes = n_semantic_classes
        self.max_scale = max_scale
        n = face_idx.shape[0]
        dev = face_idx.device
        self.register_buffer("face_idx", face_idx.long())
        self.bary_uv = nn.Parameter(bary[:, 1:3].float().clone().to(dev))
        if uv is not None and uv.numel() > 0:
            self.register_buffer("uv", uv.float())
        else:
            self.register_buffer("uv", torch.zeros(0, 2))

        if is_gum is None:
            is_gum = torch.zeros(n, dtype=torch.bool, device=dev)
        if is_h_pin is None:
            is_h_pin = torch.zeros(n, dtype=torch.bool, device=dev)
        if is_teeth is None:
            is_teeth = torch.zeros(n, dtype=torch.bool, device=dev)
        self.register_buffer("is_gum", is_gum.bool().to(dev))
        self.register_buffer("is_h_pin", is_h_pin.bool().to(dev))
        self.register_buffer("is_teeth", is_teeth.bool().to(dev))
        self.h_trainable = False

        self.register_buffer("face_semantic_class", torch.zeros(0, dtype=torch.long))
        if n_semantic_classes > 0:
            self._init_face_semantic_classes()

        self.h = nn.Parameter(torch.zeros(n, 1, device=dev))
        self.log_scale = nn.Parameter(torch.zeros(n, 3, device=dev))
        self.rotation = nn.Parameter(torch.zeros(n, 4, device=dev))
        self.rotation.data[:, 0] = 1.0
        # GaussianBlendshapes/3DGS-style low initial opacity; alpha is learned by RGB/mask.
        opacity_init = float(torch.logit(torch.tensor(0.1, device=dev)))
        self.opacity = nn.Parameter(torch.full((n, 1), opacity_init, device=dev))
        if self.sh_dim > 1:
            self.color = nn.Parameter(torch.zeros(n, self.sh_dim, 3, device=dev))
        else:
            self.color = nn.Parameter(torch.zeros(n, 3, device=dev))
        # [N, 3(out RGB), 3(in pose angles)].
        self.color_pose = nn.Parameter(torch.zeros(n, 3, 3, device=dev))
        # [N, K(=53 coeffs), 3(out RGB)].
        self.color_expression = nn.Parameter(
            torch.zeros(n, int(getattr(ict, "num_expression", 53)), 3, device=dev)
        )

        self.register_buffer(
            "anchor_vertex_ids",
            torch.tensor(list(ict.face_indices), dtype=torch.long, device=dev),
        )
        template_verts = ict.template_reference_verts()
        if template_verts.ndim == 3:
            template_verts = template_verts[0]
        self.mesh_pose = MeshGaussianPoseHelper(template_verts, ict.faces)
        self.with_mesh_scaling = bool(with_mesh_scaling)
        self.scale_max_clamp_factor = float(scale_max_clamp_factor)
        self.expression_coeff_eps = float(expression_coeff_eps)
        self.expression_support_train_mask = float(expression_support_train_mask)
        self.color_expression_exclude_mouth_eye = bool(color_expression_exclude_mouth_eye)
        self._init_color_expression_support(
            alpha=float(expression_support_alpha),
            dilate_rings=int(expression_support_dilate_rings),
            train_mask=float(expression_support_train_mask),
        )

    @property
    def bary(self):
        u = self.bary_uv[:, 0:1]
        v = self.bary_uv[:, 1:2]
        w = 1.0 - u - v
        return torch.cat([w, u, v], dim=-1)

    def _scale_clamp_max(self):
        f = getattr(self, "scale_max_clamp_factor", 0.0)
        if f and f > 0:
            return float(self.max_scale) * f
        return None

    def _effective_scale(self, mesh_verts=None, face_scale=None):
        """
        Per-Gaussian [N,3] scale as used in render: with mesh scaling
        ``exp(log_scale) * (A_pose/A_cano)`` else ``exp(log_scale)``, then optional hard cap.
        """
        if self.with_mesh_scaling:
            if face_scale is None and mesh_verts is not None:
                if mesh_verts.ndim == 3:
                    mesh_verts = mesh_verts[0]
                _, face_scale = self.mesh_pose(mesh_verts)
            if face_scale is not None:
                ratio = face_scale[self.face_idx.long()]
                scale = torch.exp(self.log_scale) * ratio
            else:
                scale = torch.exp(self.log_scale)
        else:
            scale = torch.exp(self.log_scale)
        cap = self._scale_clamp_max()
        if cap is not None:
            scale = scale.clamp(max=cap)
        return scale

    @torch.no_grad()
    def activated_scale_max(self, mesh_verts=None):
        """Per-Gaussian max axis scale used in forward (for densify / prune)."""
        return self._effective_scale(mesh_verts=mesh_verts).amax(dim=-1)

    @torch.no_grad()
    def densify_scale_max(self):
        """
        GB ``scaling_activation(_scaling).max(dim=1)`` for clone/split/world prune.

        Uses ``exp(log_scale)`` only (anisotropic weights), excluding mesh face-area ratio
        which is applied at render time via ``_effective_scale``.
        """
        return torch.exp(self.log_scale).amax(dim=-1)

    def _h_for_forward(self):
        if getattr(self, "h_trainable", False):
            return self.h
        return torch.zeros_like(self.h)

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
        k_face=4,
        k_head=4,
        k_mouth_socket=1,
        k_mouth_interior=1,
        k_teeth=1,
        k_eye_socket=1,
        k_eyeball_sclera=4,
        k_eye_occlusion=4,
        k_per_face=None,
        sh_dim=3,
        n_semantic_classes=0,
        gaussian_scale_knn_k=4,
        gaussian_scale_knn_factor=1.0,
        face_center_init=False,
        max_scale=0.008,
        with_mesh_scaling=True,
        scale_max_clamp_factor=5.0,
        expression_support_train_mask=0.25,
        color_expression_exclude_mouth_eye=False,
        **_,
    ):
        device = ict.neutral_mesh.device
        if k_per_face is not None:
            k_face = k_head = k_per_face
            k_mouth_socket = k_mouth_interior = k_eye_socket = max(1, k_per_face // 4)
            k_eyeball_sclera = k_eye_occlusion = max(1, k_per_face // 2)

        face_idx, bary, _, uv, is_gum, is_h_pin, is_teeth = build_surface_gaussian_layout(
            ict,
            ict.faces,
            k_face=k_face,
            k_head=k_head,
            k_mouth_socket=k_mouth_socket,
            k_mouth_interior=k_mouth_interior,
            k_teeth=k_teeth,
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
            is_teeth=is_teeth,
            deformer=deformer,
            sh_dim=sh_dim,
            n_semantic_classes=n_semantic_classes,
            max_scale=max_scale,
            with_mesh_scaling=with_mesh_scaling,
            scale_max_clamp_factor=scale_max_clamp_factor,
            expression_support_train_mask=expression_support_train_mask,
            color_expression_exclude_mouth_eye=color_expression_exclude_mouth_eye,
        )
        model._init_surface_region_codes()
        model._apply_color_expression_region_exclusion()
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
        with_mesh_scaling=True,
        scale_max_clamp_factor=5.0,
        expression_support_train_mask=0.25,
        color_expression_exclude_mouth_eye=False,
        n_semantic_classes=0,
    ):
        """
        Rebuild surface layout from a training checkpoint ``avatar`` state_dict
        (after densification), not from ``from_ict`` sampling counts.
        """
        from model.build import sh_dim_from_avatar_state

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

        sh_dim = sh_dim_from_avatar_state(state_dict)
        model = cls(
            ict,
            face_idx,
            bary_init,
            uv=uv,
            is_gum=state_dict["is_gum"].to(device=device),
            is_h_pin=state_dict["is_h_pin"].to(device=device),
            is_teeth=state_dict.get(
                "is_teeth",
                torch.zeros(state_dict["face_idx"].shape[0], dtype=torch.bool, device=device),
            ).to(device=device),
            deformer=deformer,
            sh_dim=sh_dim,
            n_semantic_classes=n_semantic_classes,
            max_scale=max_scale,
            with_mesh_scaling=with_mesh_scaling,
            scale_max_clamp_factor=scale_max_clamp_factor,
            expression_support_train_mask=expression_support_train_mask,
            color_expression_exclude_mouth_eye=color_expression_exclude_mouth_eye,
        )
        model._init_surface_region_codes()
        if "face_texture_map_id" in state_dict:
            model.register_buffer(
                "face_texture_map_id",
                state_dict["face_texture_map_id"].to(device=device, dtype=torch.long),
            )
        elif ict.has_texture_maps():
            tid = ict.face_texture_map_id[model.face_idx].long()
            model.register_buffer("face_texture_map_id", tid)
        model.load_avatar_state_dict(state_dict)
        return model

    @staticmethod
    def _normalize_avatar_state_dict(state_dict):
        out = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            out[key] = v
        return out

    _CRITICAL_LOAD_KEYS = (
        "color",
        "log_scale",
        "opacity",
        "rotation",
        "h",
        "bary_uv",
        "color_expression",
        "color_pose",
    )
    _REQUIRED_LOAD_KEYS = _CRITICAL_LOAD_KEYS + ("face_idx", "is_gum", "is_h_pin")
    _BUFFER_LOAD_KEYS = (
        "is_teeth",
        "uv",
        "face_region_code",
        "face_texture_map_id",
        "face_expression_support",
    )

    def load_avatar_state_dict(self, state_dict):
        """Restore avatar weights/buffers (densified layout). Raises on any critical miss/mismatch."""
        state_dict = self._normalize_avatar_state_dict(state_dict)
        missing_required = [k for k in self._REQUIRED_LOAD_KEYS if k not in state_dict]
        if missing_required:
            raise RuntimeError(
                f"avatar load: checkpoint missing required keys {missing_required} "
                f"(have {sorted(state_dict.keys())})"
            )

        for k in self._CRITICAL_LOAD_KEYS:
            if not hasattr(self, k):
                raise RuntimeError(f"avatar load: model has no parameter {k!r}")
            ckpt_shape = tuple(state_dict[k].shape)
            model_shape = tuple(getattr(self, k).shape)
            if ckpt_shape != model_shape:
                raise RuntimeError(
                    f"avatar load: {k} shape mismatch — checkpoint {ckpt_shape} vs model {model_shape}"
                )

        load_state = dict(state_dict)
        load_state.pop("template_tri_frames", None)
        load_state.pop("with_mesh_scaling", None)
        load_state.pop("color_expression_support", None)
        for k in ("sem_logits", "sem_anchor", "sem_prob_fixed", "sem_frozen_dims"):
            load_state.pop(k, None)

        incompatible = self.load_state_dict(load_state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        critical_missing = missing & set(self._CRITICAL_LOAD_KEYS)
        if critical_missing:
            raise RuntimeError(
                f"avatar load: load_state_dict missing critical keys {sorted(critical_missing)}"
            )
        if missing:
            print(f"avatar load: missing (non-critical): {sorted(missing)}")
        if unexpected:
            print(f"avatar load: ignored unexpected: {sorted(unexpected)}")

        with torch.no_grad():
            for k in self._CRITICAL_LOAD_KEYS:
                tgt = getattr(self, k)
                src = state_dict[k].to(device=tgt.device, dtype=tgt.dtype)
                tgt.copy_(src)
            self.face_idx.copy_(state_dict["face_idx"].to(self.face_idx.device, dtype=torch.long))
            for buf in self._BUFFER_LOAD_KEYS:
                if buf in state_dict and hasattr(self, buf):
                    getattr(self, buf).copy_(
                        state_dict[buf].to(getattr(self, buf).device, dtype=getattr(self, buf).dtype)
                    )

        self._verify_avatar_load(state_dict)
        return incompatible

    @torch.no_grad()
    def _verify_avatar_load(self, state_dict):
        diffs = {}
        for k in self._CRITICAL_LOAD_KEYS:
            tgt = getattr(self, k)
            src = state_dict[k].to(device=tgt.device, dtype=tgt.dtype)
            diffs[k] = float((tgt - src).abs().max().item())

        if self.color.ndim == 3:
            dc = self.color[:, 0, :]
        else:
            dc = self.color
        dc_sig = float(torch.sigmoid(dc).mean().item())
        ls_mean = float(self.log_scale.mean().item())

        parts = " ".join(f"{k}_diff={diffs[k]:.3e}" for k in self._CRITICAL_LOAD_KEYS)
        print(
            f"avatar load OK: n={self.n_gaussians} sh_dim={self.sh_dim} "
            f"dc_sigmoid_mean={dc_sig:.4f} log_scale_mean={ls_mean:.4f} {parts}"
        )
        bad = {k: v for k, v in diffs.items() if v > 1e-5}
        if bad:
            raise RuntimeError(f"avatar load: tensors did not copy — {bad}")

    def _init_knn_scales(self, k=3, scale_factor=1.0):
        from utils.gaussian_scale_init import init_module_log_scale, surface_gaussian_xyz

        xyz = surface_gaussian_xyz(self.ict, self.face_idx, self.bary, h=self.h)
        init_module_log_scale(self, xyz, k=k, scale_factor=scale_factor)

    def _init_face_semantic_classes(self):
        if self.n_semantic_classes == 0:
            return
        from rendering.gaussian_semantics import ict_face_semantic_class_table

        table = ict_face_semantic_class_table(
            self.ict.faces, self.ict, self.face_idx.device
        )
        self.register_buffer("face_semantic_class", table)

    def _semantic_features(self):
        """[N, K] one-hot from embedded ``face_idx`` (grad flows via xyz/opacity/scale, not class labels)."""
        if self.n_semantic_classes == 0 or self.face_semantic_class.numel() == 0:
            return None
        from rendering.gaussian_semantics import gaussian_semantic_onehot

        return gaussian_semantic_onehot(
            self.face_idx, self.face_semantic_class, self.n_semantic_classes
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

    def _color_expression_region_allow(self):
        if not self.color_expression_exclude_mouth_eye:
            return None
        if not hasattr(self, "face_region_code"):
            return None
        from model.color_expression_region_gate import color_expression_region_allow

        return color_expression_region_allow(
            self.face_region_code,
            enabled=True,
            dtype=self.color.dtype,
        )

    @torch.no_grad()
    def _apply_color_expression_region_exclusion(self):
        """Zero color_expression weights on mouth/eye Gaussians (base color still trains)."""
        gate = self._color_expression_region_allow()
        if gate is None:
            return
        self.color_expression.mul_(gate.unsqueeze(-1).unsqueeze(-1))

    def _init_color_expression_support(self, alpha=0.1, dilate_rings=7, train_mask=0.25):
        # Per-ICT-face AU support [F, K]; index by current face_idx at forward (densify / walk safe).
        _, support = precompute_expression_support(
            self.ict,
            alpha=alpha,
            dilate_rings=dilate_rings,
        )  # [K, V]
        dev = self.face_idx.device
        support = support.to(device=dev, dtype=self.color.dtype)
        tri_vidx = self.ict.faces.long().to(dev)  # [F, 3]
        face_support = support[:, tri_vidx].amax(dim=-1).transpose(0, 1).contiguous()  # [F, K]
        self.register_buffer("face_expression_support", face_support)
        gauss_support = face_support[self.face_idx.long()]
        mask = (gauss_support >= float(train_mask)).to(device=dev, dtype=self.color.dtype)
        with torch.no_grad():
            self.color_expression.mul_(mask.unsqueeze(-1))

    def _color_expression_support_raw(self):
        """[N, K] ICT AU support without mouth/eye region gate."""
        return self.face_expression_support[self.face_idx.long()]

    def _color_expression_support(self):
        """[N, K] support for current Gaussian face_idx embedding."""
        support = self._color_expression_support_raw()
        gate = self._color_expression_region_allow()
        if gate is not None:
            support = support * gate.unsqueeze(-1)
        return support

    @torch.no_grad()
    def validate_color_expression_shapes(self, expr_coeff=None):
        """
        Shape / indexing integrity for face-wise support and per-Gaussian color_expression.
        Raises AssertionError on mismatch.
        """
        n = self.n_gaussians
        f = int(self.ict.faces.shape[0])
        k = int(self.color_expression.shape[1])
        assert self.face_expression_support.shape == (f, k), (
            f"face_expression_support {tuple(self.face_expression_support.shape)} != ({f}, {k})"
        )
        assert self.face_idx.shape == (n,), f"face_idx {self.face_idx.shape} != ({n},)"
        assert self.color.shape == (n, 3)
        assert self.color_expression.shape == (n, k, 3)
        fidx = self.face_idx.long()
        assert int(fidx.min()) >= 0
        assert int(fidx.max()) < f, f"face_idx max {int(fidx.max())} >= num_faces {f}"
        support = self._color_expression_support()
        assert support.shape == (n, k), f"gathered support {support.shape} != ({n}, {k})"
        ref = self.face_expression_support[fidx]
        raw = self._color_expression_support_raw()
        assert torch.equal(raw, ref)
        if expr_coeff is not None:
            c = expr_coeff
            if c.ndim == 2:
                c = c.mean(dim=0)
            assert c.shape == (k,), f"expr_coeff {c.shape} != ({k},)"
        return dict(n_gaussians=n, n_faces=f, n_expression=k, support=support)

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
        enable_color_pose=False,
        enable_color_expression=True,
    ):
        if verts.ndim == 3:
            verts = verts[0]
        bary = self.bary
        xyz_base = sample_surface(verts, faces, self.face_idx, bary)
        vn = sample_normals(
            self._vertex_normals(verts, faces), faces, self.face_idx, bary
        )
        h_eff = self._h_for_forward()
        xyz = xyz_base + h_eff * vn
        per_vert_q, face_scale = self.mesh_pose(verts)
        mesh_quat = barycentric_vertex_quaternion(
            per_vert_q, self.mesh_pose.faces, self.face_idx, bary
        )
        local_quat = _normalize_quat_wxyz(self.rotation)
        rotation = _normalize_quat_wxyz(quaternion_multiply(mesh_quat, local_quat))
        scale = self._effective_scale(face_scale=face_scale)
        opacity = torch.sigmoid(self.opacity)
        color = self.color
        color_delta = 0.0
        if enable_color_pose and hasattr(self, "color_pose"):
            if pose_angle_vec is None:
                pose_angle_vec = torch.zeros(
                    self.face_idx.shape[0], 3, device=self.face_idx.device, dtype=self.color.dtype
                )
            color_delta = color_delta + torch.einsum("nij,nj->ni", self.color_pose, pose_angle_vec)
        if enable_color_expression and hasattr(self, "color_expression"):
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
            support = self._color_expression_support()
            effective = torch.abs(expr_coeff) * support.amax(dim=0)
            active = torch.nonzero(effective > self.expression_coeff_eps, as_tuple=False).squeeze(-1)
            if active.numel() > 0:
                train_mask = (support[:, active] >= self.expression_support_train_mask).to(
                    dtype=self.color.dtype
                )
                color_expr = self.color_expression[:, active, :] * train_mask.unsqueeze(-1)
                coeff_active = expr_coeff[active]
                support_active = support[:, active] * train_mask
                color_delta = color_delta + torch.einsum(
                    "nkr,nk,k->nr",
                    color_expr,
                    support_active,
                    coeff_active,
                )

        if isinstance(color_delta, torch.Tensor):
            if self.sh_dim > 1:
                # Add expression/pose delta to the DC component
                color_dc = color[:, 0, :] + color_delta
                color = torch.cat([color_dc.unsqueeze(1), color[:, 1:, :]], dim=1)
            else:
                color = color + color_delta

        out = {
            "xyz": xyz,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity,
            "color": color,
            "h": self._h_for_forward(),
            "face_idx": self.face_idx,
            "bary": bary,
            "normals": vn,
            "group": "surface",
            "is_h_pin": self.is_h_pin,
        }
        sem_feat = self._semantic_features()
        if sem_feat is not None:
            out["sem_features"] = sem_feat
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
        enable_color_pose=False,
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
                pose_translation_global=tracker_out.get("translation_global"),
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
        if surface_out.get("sem_features") is not None:
            out["sem_features"] = surface_out["sem_features"]
        return out
