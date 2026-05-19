"""
Bilateral eye Gaussians: **one** shared chart UV + appearance; per-eye gaze slide + mesh instantiate.

Left and right use the same ``uv``, ``color``, ``log_scale``, ``rotation``, ``opacity``.
Shared chart ``uv`` + per-side gaze slide. Index ``i`` uses the same ``uv[i]`` and Gaussian
params; ``(face_idx, bary)`` are resolved on each eye's sclera chart (global face id differs L/R).
"""

import torch
import torch.nn as nn

from rendering.gaussian_semantics import eye_fixed_semantic_probs
from utils.barycentric import sample_normals, sample_surface
from utils.eye_chart import embed_chart_uv_on_mesh
from utils.eye_uv_sampling import sample_shared_sclera_layout
from utils.gaze_uv import combine_gaze
from utils.mesh_ops import vertex_normals
from utils.uv_mesh import UVMesh

# Pentagon in chart-local UV around disk center (0.5, 0.5).
IRIS_CONTROL_TEMPLATE_UV = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 0.04],
        [0.04, 0.0],
        [0.0, -0.04],
        [-0.04, 0.0],
    ],
    dtype=torch.float32,
)

SCLERA_CHART_CENTER_UV = torch.tensor([0.5, 0.5], dtype=torch.float32)


def _freeze_attr_as_buffer(module: nn.Module, name: str):
    if not hasattr(module, name):
        return

    value = getattr(module, name)
    if value is None:
        return

    value = value.detach().clone()

    if name in module._parameters:
        del module._parameters[name]
    elif name in module._buffers:
        module._buffers[name] = value
        return
    else:
        delattr(module, name)

    module.register_buffer(name, value)


class EyeTextureGaussians(nn.Module):
    """
    Shared sclera chart parameters for both eyes.

    Per side at forward: ``uv_eff = uv + gaze`` (mirror U on R) → chart→mesh embed → posed xyz.
    """

    def __init__(
        self,
        n_per_eye=1024,
        sh_dim=3,
        gaze_uv_range=0.12,
        learn_gaze_refine=True,
        n_semantic_classes=7,
        mirror_right_u=False,
        ict=None,
        n_iris_control=5,
        fixed_h=0.0,
        uv_sample_mode="hemisphere",
        sclera_min_front_dot=0.0,
        sclera_hemisphere_only=True,
    ):
        super().__init__()
        self.n_per_eye = n_per_eye
        self.n_iris_control = min(n_iris_control, 5)
        self.gaze_uv_range = gaze_uv_range
        self.n_semantic_classes = n_semantic_classes
        self.mirror_right_u = bool(mirror_right_u)
        self.fixed_h = fixed_h
        self.sclera_min_front_dot = float(sclera_min_front_dot)
        self.sclera_hemisphere_only = bool(sclera_hemisphere_only)

        device = ict.neutral_mesh.device if ict is not None else "cpu"
        self._ict_for_scale_init = ict

        if ict is not None:
            uv_init, _, _, _, _ = sample_shared_sclera_layout(
                ict,
                n_per_eye,
                device,
                min_front_dot=sclera_min_front_dot,
                hemisphere_only=sclera_hemisphere_only,
                mode=uv_sample_mode,
                mirror_right_u=self.mirror_right_u,
            )
            pole = SCLERA_CHART_CENTER_UV.to(device)
            tpl = IRIS_CONTROL_TEMPLATE_UV[: self.n_iris_control].to(pole.device)
            iris_control_uv = pole.unsqueeze(0) + tpl
            self.register_buffer("iris_control_uv", iris_control_uv)
        else:
            uv_init = torch.rand(n_per_eye, 2)
            self.register_buffer(
                "iris_control_uv",
                SCLERA_CHART_CENTER_UV.unsqueeze(0)
                + IRIS_CONTROL_TEMPLATE_UV[: self.n_iris_control],
            )

        self.uv = nn.Parameter(uv_init.clone())
        self.register_buffer("h", torch.full((n_per_eye, 1), float(fixed_h)))

        self.sem_logits = None
        self.register_buffer("sem_prob_fixed", None)
        if n_semantic_classes > 0:
            eye_sem = eye_fixed_semantic_probs(n_per_eye, n_semantic_classes, device="cpu")
            self.register_buffer("sem_prob_fixed", eye_sem)

        self.log_scale = nn.Parameter(torch.zeros(n_per_eye, 3))
        self.rotation = nn.Parameter(torch.zeros(n_per_eye, 4))
        self.rotation.data[:, 0] = 1.0
        self.opacity = nn.Parameter(torch.zeros(n_per_eye, 1))
        self.color = nn.Parameter(torch.zeros(n_per_eye, sh_dim))

        with torch.no_grad():
            self.h.zero_()

        _freeze_attr_as_buffer(self, "uv")
        _freeze_attr_as_buffer(self, "h")

        if learn_gaze_refine:
            self.gaze_refine_left = nn.Parameter(torch.zeros(2))
            self.gaze_refine_right = nn.Parameter(torch.zeros(2))
        else:
            self.gaze_refine_left = None
            self.gaze_refine_right = None

    def init_log_scale_from_mesh(self, ict, k=3, scale_factor=1.0):
        from utils.gaussian_scale_init import init_module_log_scale

        ict = ict if ict is not None else self._ict_for_scale_init
        if ict is None:
            return
        if hasattr(ict, "canonical") and ict.canonical is not None:
            verts = ict.canonical[0]
        else:
            verts = ict.neutral_mesh[0]
        device = self.uv.device
        faces = ict.faces.to(device)
        with torch.no_grad():
            gaze0 = torch.zeros(2, device=device, dtype=self.uv.dtype)
            fi, bary = self._instantiate_mesh(ict, "L", gaze0)
            xyz = sample_surface(verts, faces, fi, bary)
            init_module_log_scale(self, xyz, k=k, scale_factor=scale_factor)

    def _apply_gaze_refine(self, gaze_uv, side):
        refine = self.gaze_refine_left if side == "L" else self.gaze_refine_right
        if refine is None:
            return gaze_uv
        if gaze_uv.ndim == 1:
            gaze_uv = gaze_uv.unsqueeze(0)
        out = combine_gaze(gaze_uv, refine.unsqueeze(0), self.gaze_uv_range)
        return out.squeeze(0) if out.shape[0] == 1 else out

    def _effective_chart_uv(self, uv_base, gaze_offset, side):
        """Shared chart UV + per-side gaze; mirror U on R when ``mirror_right_u``."""
        uv_eff = uv_base + gaze_offset.reshape(1, 2)
        if side == "R" and self.mirror_right_u:
            uv_eff = torch.stack([1.0 - uv_eff[:, 0], uv_eff[:, 1]], dim=-1)
        return uv_eff

    def _instantiate_mesh(self, ict, side, gaze_offset):
        """Shared chart UV + gaze → this eye's global ``(face_idx, bary)``."""
        uv_eff = self._effective_chart_uv(self.uv.detach(), gaze_offset, side)
        return embed_chart_uv_on_mesh(
            ict,
            side,
            uv_eff,
            uv_eff.device,
            mirror_right_u=False,
            min_front_dot=self.sclera_min_front_dot,
            hemisphere_only=self.sclera_hemisphere_only,
        )

    def _mesh_verts(self, verts, uv_mesh: UVMesh):
        if verts is None:
            return uv_mesh.verts
        if verts.ndim == 3:
            return verts[0]
        return verts

    def _iris_control_xyz(self, ict, uv_mesh: UVMesh, gaze_offset, side: str, verts):
        if self.n_iris_control <= 0 or ict is None:
            return None
        mesh_v = self._mesh_verts(verts, uv_mesh)
        device = mesh_v.device
        ctrl_eff = self._effective_chart_uv(self.iris_control_uv.to(device), gaze_offset, side)
        fi, bary = embed_chart_uv_on_mesh(
            ict,
            side,
            ctrl_eff,
            device,
            mirror_right_u=False,
            min_front_dot=self.sclera_min_front_dot,
            hemisphere_only=self.sclera_hemisphere_only,
        )
        return sample_surface(mesh_v, uv_mesh.faces, fi, bary)

    def _forward_one(
        self,
        ict,
        uv_mesh: UVMesh,
        gaze_offset,
        side: str,
        verts=None,
    ):
        device = self.uv.device
        uv_base = self.uv.detach()
        uv_eff = self._effective_chart_uv(uv_base, gaze_offset, side)

        face_idx, bary = self._instantiate_mesh(ict, side, gaze_offset)

        mesh_v = self._mesh_verts(verts, uv_mesh)
        faces_m = uv_mesh.faces
        xyz = sample_surface(mesh_v, faces_m, face_idx, bary)
        vn = vertex_normals(mesh_v, faces_m)
        normals = sample_normals(vn, faces_m, face_idx, bary)

        h = torch.zeros(uv_eff.shape[0], 1, dtype=uv_eff.dtype, device=device)
        scale = torch.exp(self.log_scale).clamp(max=0.02)
        opacity = torch.sigmoid(self.opacity)

        out = {
            "xyz": xyz,
            "scale": scale,
            "rotation": self.rotation,
            "opacity": opacity,
            "color": self.color,
            "h": h,
            "uv": uv_eff,
            "uv_base": uv_base,
            "gaze_offset": gaze_offset,
            "face_idx": face_idx,
            "bary": bary,
            "normals": normals,
            "side": side,
        }

        if self.sem_prob_fixed is not None:
            out["sem_prob"] = self.sem_prob_fixed
        elif self.sem_logits is not None:
            out["sem_prob"] = torch.softmax(self.sem_logits, dim=-1)

        return out

    def forward(
        self,
        left_uv_mesh: UVMesh,
        right_uv_mesh: UVMesh,
        verts=None,
        faces=None,
        gaze_uv_left=None,
        gaze_uv_right=None,
    ):
        ict = self._ict_for_scale_init
        device = self.uv.device
        dtype = self.uv.dtype

        if gaze_uv_left is None:
            gaze_uv_left = torch.zeros(2, device=device, dtype=dtype)
        else:
            gaze_uv_left = torch.as_tensor(gaze_uv_left, device=device, dtype=dtype)

        if gaze_uv_right is None:
            gaze_uv_right = torch.zeros(2, device=device, dtype=dtype)
        else:
            gaze_uv_right = torch.as_tensor(gaze_uv_right, device=device, dtype=dtype)

        if gaze_uv_left.ndim == 2:
            gaze_l = self._apply_gaze_refine(gaze_uv_left[0], "L")
            gaze_r = self._apply_gaze_refine(gaze_uv_right[0], "R")
        else:
            gaze_l = self._apply_gaze_refine(gaze_uv_left, "L")
            gaze_r = self._apply_gaze_refine(gaze_uv_right, "R")

        out_l = self._forward_one(ict, left_uv_mesh, gaze_l, side="L", verts=verts)
        out_r = self._forward_one(ict, right_uv_mesh, gaze_r, side="R", verts=verts)

        xyz = torch.cat([out_l["xyz"], out_r["xyz"]], dim=0)
        iris_l = self._iris_control_xyz(ict, left_uv_mesh, gaze_l, "L", verts)
        iris_r = self._iris_control_xyz(ict, right_uv_mesh, gaze_r, "R", verts)
        if iris_l is not None and iris_r is not None:
            iris_control_xyz = torch.cat([iris_l, iris_r], dim=0)
        else:
            iris_control_xyz = None

        out = {
            "left": out_l,
            "right": out_r,
            "xyz": xyz,
            "scale": torch.cat([out_l["scale"], out_r["scale"]], dim=0),
            "rotation": torch.cat([out_l["rotation"], out_r["rotation"]], dim=0),
            "opacity": torch.cat([out_l["opacity"], out_r["opacity"]], dim=0),
            "color": torch.cat([out_l["color"], out_r["color"]], dim=0),
            "h": torch.cat([out_l["h"], out_r["h"]], dim=0),
            "uv": torch.cat([out_l["uv"], out_r["uv"]], dim=0),
            "iris_control_xyz": iris_control_xyz,
            "is_eyeball_surface": torch.ones(xyz.shape[0], dtype=torch.bool, device=device),
            "shared_uv_base": self.uv,
            "gaze_offset_left": gaze_l,
            "gaze_offset_right": gaze_r,
            "mirror_right_u": self.mirror_right_u,
        }

        if out_l.get("sem_prob") is not None:
            out["sem_prob"] = torch.cat([out_l["sem_prob"], out_r["sem_prob"]], dim=0)

        return out
