"""MediaPipe correction: split MLPs for expression / pose / gaze."""

import torch
import torch.nn as nn

from utils.gaze_uv import GazeCalibrator, base_gaze_from_mediapipe, combine_gaze
from utils.smoothstep import smoothstep
from utils.tracker_inputs import (
    GAZE_FACE_DIR_MP,
    GAZE_IRIS_MP,
    POSE_LMK_MP,
    gather_mp_landmarks,
)


def _mlp(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
    )


class TrackerCorrectionMLP(nn.Module):
    """
    Three trunks (no shared hidden state):

    - **expr**: blendshapes + optional full 478×2 landmarks → gamma
    - **pose**: raw 6D pose + sparse head anchors → rotation(6) + translation(3) residuals.
      Optional global ``log_pose_scale`` (training off by default; mesh uses scale=1).
    - **gaze**: raw pose + face-direction landmarks + iris 468–477 → gaze UV residual
    """

    def __init__(
        self,
        n_blendshapes=52,
        hidden=128,
        hidden_pose=64,
        hidden_gaze=64,
        pose_dim=6,
        gamma_min=0.4,
        gamma_max=2.5,
        active_lo=0.02,
        active_hi=0.08,
        gaze_uv_range=0.12,
        use_landmarks=True,
    ):
        super().__init__()
        self.n_blendshapes = n_blendshapes
        self.pose_dim = pose_dim
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.active_lo = active_lo
        self.active_hi = active_hi
        self.gaze_uv_range = gaze_uv_range
        self.use_landmarks = use_landmarks
        self.gaze_calibrator = GazeCalibrator(uv_range=gaze_uv_range)

        self.log_pose_scale = nn.Parameter(torch.zeros(1))

        expr_in = n_blendshapes + (478 * 2 if use_landmarks else 0)
        self.expr_trunk = _mlp(expr_in, hidden)
        self.head_gamma = nn.Linear(hidden, n_blendshapes)

        pose_in = pose_dim + len(POSE_LMK_MP) * 2
        self.pose_trunk = _mlp(pose_in, hidden_pose)
        self.head_pose = nn.Linear(hidden_pose, 9)

        gaze_in = pose_dim + len(GAZE_FACE_DIR_MP) * 2 + len(GAZE_IRIS_MP) * 2
        self.gaze_trunk = _mlp(gaze_in, hidden_gaze)
        self.head_gaze_l = nn.Linear(hidden_gaze, 2)
        self.head_gaze_r = nn.Linear(hidden_gaze, 2)

    @property
    def pose_scale(self):
        return torch.exp(self.log_pose_scale)

    def forward(
        self,
        mp_blendshape,
        mp_landmarks_2d=None,
        mp_pose_raw=None,
        force_gamma_one=False,
    ):
        B = mp_blendshape.shape[0]
        device = mp_blendshape.device
        dtype = mp_blendshape.dtype

        if mp_pose_raw is None:
            mp_pose_raw = torch.zeros(B, self.pose_dim, device=device, dtype=dtype)
        if mp_pose_raw.shape[-1] == 16:
            mp_pose_raw = mp_pose_raw[..., :6]

        c_raw = mp_blendshape.clamp(1e-6, 1.0)
        active = smoothstep(c_raw, self.active_lo, self.active_hi).detach()

        # --- expression / gamma ---
        expr_parts = [mp_blendshape]
        if self.use_landmarks and mp_landmarks_2d is not None:
            expr_parts.append(mp_landmarks_2d.reshape(B, -1))
        h_expr = self.expr_trunk(torch.cat(expr_parts, dim=-1))
        raw_gamma = self.head_gamma(h_expr)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw_gamma)
        if force_gamma_one:
            gamma = torch.ones_like(gamma)
        coeffs_corr = active * c_raw.pow(gamma)

        # --- pose: global scale + rot/trans residual ---
        pose_lmk = gather_mp_landmarks(mp_landmarks_2d, POSE_LMK_MP, device, dtype)
        h_pose = self.pose_trunk(torch.cat([mp_pose_raw, pose_lmk], dim=-1))
        pose_delta = self.head_pose(h_pose)
        pose_residual = pose_delta[:, :6]
        translation_residual = pose_delta[:, 6:9]
        scale = self.pose_scale.expand(B)

        # --- gaze ---
        gaze_face = gather_mp_landmarks(mp_landmarks_2d, GAZE_FACE_DIR_MP, device, dtype)
        gaze_iris = gather_mp_landmarks(mp_landmarks_2d, GAZE_IRIS_MP, device, dtype)
        h_gaze = self.gaze_trunk(torch.cat([mp_pose_raw, gaze_face, gaze_iris], dim=-1))

        gaze_base_l, gaze_base_r = base_gaze_from_mediapipe(c_raw, self.gaze_calibrator)
        gaze_res_l = self.head_gaze_l(h_gaze)
        gaze_res_r = self.head_gaze_r(h_gaze)
        gaze_uv_left = combine_gaze(gaze_base_l, gaze_res_l, self.gaze_uv_range)
        gaze_uv_right = combine_gaze(gaze_base_r, gaze_res_r, self.gaze_uv_range)

        return {
            "gamma": gamma,
            "coeffs": coeffs_corr,
            "coeffs_raw": c_raw,
            "active": active,
            "gaze_base_left": gaze_base_l,
            "gaze_base_right": gaze_base_r,
            "gaze_residual_left": gaze_res_l,
            "gaze_residual_right": gaze_res_r,
            "gaze_uv_left": gaze_uv_left,
            "gaze_uv_right": gaze_uv_right,
            "pose_scale": scale,
            "pose_residual": pose_residual,
            "translation_residual": translation_residual,
        }
