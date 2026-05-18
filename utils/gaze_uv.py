"""MediaPipe eyeLook linear gaze prior + optional learnable calibration."""

import torch
import torch.nn as nn

from utils.mediapipe_blendshapes import (
    MP_EYE_LOOK_DOWN_L,
    MP_EYE_LOOK_DOWN_R,
    MP_EYE_LOOK_IN_L,
    MP_EYE_LOOK_IN_R,
    MP_EYE_LOOK_OUT_L,
    MP_EYE_LOOK_OUT_R,
    MP_EYE_LOOK_UP_L,
    MP_EYE_LOOK_UP_R,
)


class GazeCalibrator(nn.Module):
    """
    Per-eye sign/scale on MediaPipe eyeLook deltas.
    Initialized so base gaze ≈ 0.75 * gaze_uv_range at unit activation.
    """

    def __init__(self, uv_range=0.12, init_scale=0.75):
        super().__init__()
        self.uv_range = uv_range
        self.sign_x_l = nn.Parameter(torch.tensor(1.0))
        self.sign_y_l = nn.Parameter(torch.tensor(1.0))
        self.sign_x_r = nn.Parameter(torch.tensor(1.0))
        self.sign_y_r = nn.Parameter(torch.tensor(1.0))
        self.scale_l = nn.Parameter(torch.tensor(float(init_scale)))
        self.scale_r = nn.Parameter(torch.tensor(float(init_scale)))

    def apply_side(self, dx, dy, sign_x, sign_y, scale):
        du = sign_x * dx
        dv = sign_y * dy
        return torch.stack([du, dv], dim=-1) * (scale * self.uv_range)


def base_gaze_from_mediapipe(mp_blendshape, calibrator: GazeCalibrator):
    """
    mp_blendshape: [B, 52] in [0, 1]
    Returns gaze_uv_left [B, 2], gaze_uv_right [B, 2]
    """
    c = mp_blendshape.clamp(0.0, 1.0)
    left_dx = c[:, MP_EYE_LOOK_OUT_L] - c[:, MP_EYE_LOOK_IN_L]
    left_dy = c[:, MP_EYE_LOOK_UP_L] - c[:, MP_EYE_LOOK_DOWN_L]
    right_dx = c[:, MP_EYE_LOOK_OUT_R] - c[:, MP_EYE_LOOK_IN_R]
    right_dy = c[:, MP_EYE_LOOK_UP_R] - c[:, MP_EYE_LOOK_DOWN_R]

    gaze_l = calibrator.apply_side(
        left_dx, left_dy, calibrator.sign_x_l, calibrator.sign_y_l, calibrator.scale_l
    )
    gaze_r = calibrator.apply_side(
        right_dx, right_dy, calibrator.sign_x_r, calibrator.sign_y_r, calibrator.scale_r
    )
    r = calibrator.uv_range
    return gaze_l.clamp(-r, r), gaze_r.clamp(-r, r)


def combine_gaze(base, residual, uv_range):
    if residual is None:
        return base.clamp(-uv_range, uv_range)
    return (base + residual).clamp(-uv_range, uv_range)


def gaze_residual_prior_loss(residual):
    if residual is None:
        return residual
    return residual.pow(2).mean()
