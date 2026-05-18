"""
Iris/sclera Gaussians on a local eye tangent plane (slide, no eyeball mesh rotation).

Gaze is encoded as per-eye UV offset; eyelid/socket deformation stays on ICT mesh.
"""

import torch
import torch.nn as nn

from utils.eye_frame import EyeFrame, points_on_eye_plane

LEFT_IRIS_MP = [468, 469, 470, 471, 472]
RIGHT_IRIS_MP = [473, 474, 475, 476, 477]

# canonical 5-point layout on local eye plane (center, top, right, bottom, left)
IRIS_TEMPLATE_UV = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 0.35],
        [0.35, 0.0],
        [0.0, -0.35],
        [-0.35, 0.0],
    ],
    dtype=torch.float32,
)


class EyePlaneGaussians(nn.Module):
    def __init__(self, n_per_eye=64, sh_dim=3, n_iris_control=5):
        super().__init__()
        self.n_per_eye = n_per_eye
        self.n_iris_control = min(n_iris_control, n_per_eye)
        n_total = n_per_eye * 2

        init_uv = torch.cat(
            [
                torch.rand(n_per_eye, 2) * 0.4 - 0.2,
                torch.rand(n_per_eye, 2) * 0.4 - 0.2,
            ],
            dim=0,
        )
        with torch.no_grad():
            init_uv[:n_iris_control] = IRIS_TEMPLATE_UV[: self.n_iris_control]
            init_uv[n_per_eye : n_per_eye + self.n_iris_control] = IRIS_TEMPLATE_UV[
                : self.n_iris_control
            ]

        self.local_uv = nn.Parameter(init_uv)
        self.h = nn.Parameter(torch.zeros(n_total, 1))

        self.log_scale = nn.Parameter(torch.full((n_total, 3), -3.0))
        self.rotation = nn.Parameter(torch.zeros(n_total, 4))
        self.opacity = nn.Parameter(torch.zeros(n_total, 1))
        self.color = nn.Parameter(torch.zeros(n_total, sh_dim))

        # per-batch gaze slide (set before forward or passed in)
        self.gaze_offset_left = None
        self.gaze_offset_right = None

    def set_gaze_offset(self, left=None, right=None):
        """left/right: [B, 2] or [2] UV displacement on eye plane."""
        self.gaze_offset_left = left
        self.gaze_offset_right = right

    def forward(self, left_frame: EyeFrame, right_frame: EyeFrame):
        n = self.n_per_eye
        uv_l = self.local_uv[:n]
        uv_r = self.local_uv[n:]
        h_l = self.h[:n]
        h_r = self.h[n:]

        xyz_l, uv_l_used = points_on_eye_plane(uv_l, h_l, left_frame, self.gaze_offset_left)
        xyz_r, uv_r_used = points_on_eye_plane(uv_r, h_r, right_frame, self.gaze_offset_right)

        xyz = torch.cat([xyz_l, xyz_r], dim=0)
        scale = torch.exp(self.log_scale).clamp(max=0.02)
        opacity = torch.sigmoid(self.opacity)

        iris_control_idx = torch.cat(
            [
                torch.arange(self.n_iris_control, device=xyz.device),
                torch.arange(n, n + self.n_iris_control, device=xyz.device),
            ]
        )

        return {
            "xyz": xyz,
            "scale": scale,
            "rotation": self.rotation,
            "opacity": opacity,
            "color": self.color,
            "h": self.h,
            "uv_left": uv_l_used,
            "uv_right": uv_r_used,
            "iris_control_xyz": xyz[iris_control_idx],
            "iris_control_idx": iris_control_idx,
            "is_anchor_surface": torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device),
            "group": "eye_plane",
        }
