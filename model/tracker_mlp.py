"""MediaPipe-only correction MLP (activation gate + gamma + pose/gaze residuals)."""

import torch
import torch.nn as nn

from utils.smoothstep import smoothstep


class TrackerCorrectionMLP(nn.Module):
    """
    MediaPipe activation pattern is preserved (detached gate).
    Gamma personalizes intensity only on active AUs.

    C_eff = active(C_raw) * C_raw ** gamma
    """

    def __init__(
        self,
        n_blendshapes=52,
        hidden=128,
        pose_dim=6,
        gamma_min=0.4,
        gamma_max=2.5,
        active_lo=0.02,
        active_hi=0.08,
        use_landmarks=False,
    ):
        super().__init__()
        self.n_blendshapes = n_blendshapes
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.active_lo = active_lo
        self.active_hi = active_hi
        self.use_landmarks = use_landmarks

        in_dim = n_blendshapes + (478 * 2 if use_landmarks else 0) + pose_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.head_gamma = nn.Linear(hidden, n_blendshapes)
        self.head_pose = nn.Linear(hidden, 6)
        self.head_trans = nn.Linear(hidden, 3)
        self.head_gaze_l = nn.Linear(hidden, 2)
        self.head_gaze_r = nn.Linear(hidden, 2)

    def forward(
        self,
        mp_blendshape,
        mp_landmarks_2d=None,
        mp_pose_raw=None,
        force_gamma_one=False,
    ):
        B = mp_blendshape.shape[0]
        if mp_pose_raw is None:
            mp_pose_raw = torch.zeros(B, 6, device=mp_blendshape.device, dtype=mp_blendshape.dtype)
        if mp_pose_raw.shape[-1] == 16:
            mp_pose_raw = mp_pose_raw[..., :6]

        parts = [mp_blendshape, mp_pose_raw]
        if self.use_landmarks and mp_landmarks_2d is not None:
            parts.append(mp_landmarks_2d.reshape(B, -1))
        x = torch.cat(parts, dim=-1)

        h = self.trunk(x)
        raw_gamma = self.head_gamma(h)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw_gamma)

        c_raw = mp_blendshape.clamp(1e-6, 1.0)
        active = smoothstep(c_raw, self.active_lo, self.active_hi).detach()

        if force_gamma_one:
            gamma = torch.ones_like(gamma)
        coeffs_corr = active * c_raw.pow(gamma)

        return {
            "gamma": gamma,
            "coeffs": coeffs_corr,
            "coeffs_raw": c_raw,
            "active": active,
            "pose_residual": self.head_pose(h),
            "translation_residual": self.head_trans(h),
            "gaze_uv_left": self.head_gaze_l(h),
            "gaze_uv_right": self.head_gaze_r(h),
        }
