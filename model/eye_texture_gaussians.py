"""
Bilateral eye Gaussians on front ``M_Sclera*`` ∩ eyeball (filled UV disk, pole-facing hemisphere+).

``M_Iris*`` charts are annuli with an empty center — not used for sampling or gaze anchors.
Iris MP landmarks are projected onto sclera triangles during bake (see eye_transplant).
"""

import torch
import torch.nn as nn

from rendering.gaussian_semantics import eye_fixed_semantic_probs
from utils.eye_uv_sampling import sample_sclera_uv
from utils.gaze_uv import combine_gaze
from utils.uv_mesh import UVMesh, surface_points_from_uvh

# Small pentagon in sclera *local* chart around chart center (pupil on M_Sclera* disk).
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
    Shared sclera UV parameters for both eyes; gaze offsets slide UV per side.

    Appearance (color/opacity) is shared; ``h`` is fixed at 0 on the eyeball chart.
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
    ):
        super().__init__()
        self.n_per_eye = n_per_eye
        self.n_iris_control = min(n_iris_control, 5)
        self.gaze_uv_range = gaze_uv_range
        self.n_semantic_classes = n_semantic_classes
        self.mirror_right_u = bool(mirror_right_u)
        self.fixed_h = fixed_h

        device = ict.neutral_mesh.device if ict is not None else "cpu"
        if ict is not None:
            from utils.eye_chart import sclera_pole_uv

            uv_init = sample_sclera_uv(ict, "L", n_per_eye, device)
            pole = torch.tensor(sclera_pole_uv(ict, "L"), dtype=torch.float32)
            tpl = IRIS_CONTROL_TEMPLATE_UV[: self.n_iris_control].to(pole.device)
            iris_control_uv = pole.unsqueeze(0) + tpl
            self.register_buffer("iris_control_uv", iris_control_uv)
        else:
            uv_init = torch.rand(n_per_eye, 2)
            self.register_buffer(
                "iris_control_uv",
                IRIS_CONTROL_TEMPLATE_UV[: self.n_iris_control].clone(),
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

    def _apply_gaze_refine(self, gaze_uv, side):
        refine = self.gaze_refine_left if side == "L" else self.gaze_refine_right
        if refine is None:
            return gaze_uv
        if gaze_uv.ndim == 1:
            gaze_uv = gaze_uv.unsqueeze(0)
        out = combine_gaze(gaze_uv, refine.unsqueeze(0), self.gaze_uv_range)
        return out.squeeze(0) if out.shape[0] == 1 else out

    def _iris_control_xyz(self, uv_mesh: UVMesh, gaze_offset, side: str, verts, faces):
        if self.n_iris_control <= 0:
            return None
        ctrl_uv = self.iris_control_uv + gaze_offset.unsqueeze(0)
        if side == "R" and self.mirror_right_u:
            ctrl_uv = torch.stack([1.0 - ctrl_uv[:, 0], ctrl_uv[:, 1]], dim=-1)
        h0 = torch.zeros(self.n_iris_control, 1, device=ctrl_uv.device, dtype=ctrl_uv.dtype)
        xyz, _, _, _ = surface_points_from_uvh(ctrl_uv, h0, uv_mesh, None)
        return xyz

    def _forward_one(self, uv_mesh: UVMesh, gaze_offset, side: str, verts=None, faces=None):
        uv_base = self.uv.detach()
        uv_eff = uv_base + gaze_offset.unsqueeze(0)

        if side == "R" and self.mirror_right_u:
            uv_eff = torch.stack([1.0 - uv_eff[:, 0], uv_eff[:, 1]], dim=-1)

        h = torch.zeros(uv_eff.shape[0], 1, dtype=uv_eff.dtype, device=uv_eff.device)

        xyz, face_idx, bary, normals = surface_points_from_uvh(uv_eff, h, uv_mesh, self)

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

        out_l = self._forward_one(left_uv_mesh, gaze_l, side="L", verts=verts, faces=faces)
        out_r = self._forward_one(right_uv_mesh, gaze_r, side="R", verts=verts, faces=faces)

        xyz = torch.cat([out_l["xyz"], out_r["xyz"]], dim=0)
        iris_l = self._iris_control_xyz(left_uv_mesh, gaze_l, "L", verts, faces)
        iris_r = self._iris_control_xyz(right_uv_mesh, gaze_r, "R", verts, faces)
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
