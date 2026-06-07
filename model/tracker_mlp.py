"""MediaPipe correction: MP→ICT FaceKit gather, then per-slot gamma + pose residuals."""

import math

import torch
import torch.nn as nn

from utils.mesh_ops import rotation_matrix_to_6d, rotation_6d_to_matrix
from utils.tracker import (
    POSE_LMK_MP,
    gather_mp_landmarks,
    landmarks_2d_canonical,
    landmarks_3d_to_camera_xy,
)


def _mlp(in_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
    )


def _init_linear_small(linear: nn.Linear, std: float = 1e-3):
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    nn.init.zeros_(linear.bias)


def _init_head_zero(linear: nn.Linear):
    """Residual head: zero W,b so trunk output starts as exact zero delta."""
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


class TrackerCorrectionMLP(nn.Module):
    """
    Two trunks (no shared hidden state):

    - **expr**: MP [B, 52] → gather → gamma (pow or additive residual) → ICT expression weights
    - **pose**: MP rotation 6D (detector) + MLP Δ_rot; local Δ_t (pose-weighted) + global Δ_t (full mesh)

    ``pose_residual`` = ``mp_rotation_6d`` + ``Δ_rot`` (Zhou 6D, MP base detached).
    ``mediapipe_to_ict`` length 53.
    """

    def __init__(
        self,
        n_blendshapes=52,
        num_ict_expression=53,
        hidden=128,
        hidden_pose=64,
        pose_dim=6,
        gamma_min=0.4,
        gamma_max=2.5,
        gamma_symmetric_log=True,
        use_landmarks=True,
        canonicalize_landmarks_2d=True,
        mediapipe_to_ict=None,
        additive_gamma_correction=False,
        **_,
    ):
        super().__init__()
        self.n_blendshapes = int(n_blendshapes)
        self.num_ict_expression = int(num_ict_expression)
        self.pose_dim = pose_dim
        self.gamma_symmetric_log = bool(gamma_symmetric_log)
        gmax = float(gamma_max)
        if self.gamma_symmetric_log:
            self.gamma_log_span = math.log(max(gmax, 1.0 + 1e-6))
            self.gamma_min = 1.0 / math.exp(self.gamma_log_span)
            self.gamma_max = math.exp(self.gamma_log_span)
        else:
            self.gamma_log_span = None
            self.gamma_min = float(gamma_min)
            self.gamma_max = gmax
        self.use_landmarks = use_landmarks
        self.canonicalize_landmarks_2d = canonicalize_landmarks_2d
        self.additive_gamma_correction = bool(additive_gamma_correction)

        if mediapipe_to_ict is not None:
            self.register_buffer(
                "mediapipe_to_ict",
                mediapipe_to_ict.detach().long().reshape(-1),
            )

        expr_in = self.n_blendshapes + (478 * 2 if use_landmarks else 0)
        self.expr_trunk = _mlp(expr_in, hidden)
        self.head_gamma = nn.Linear(hidden, self.num_ict_expression)

        pose_in = pose_dim + len(POSE_LMK_MP) * 2
        self.pose_trunk = _mlp(pose_in, hidden_pose)
        self.head_pose = nn.Linear(hidden_pose, 12)
        self.register_buffer(
            "identity_6d",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        )
        self.log_pose_scale = nn.Parameter(torch.zeros(1))
        # Subject-global translation (no_gamma_and_pose only); not per-frame network output.
        self.global_translation = nn.Parameter(torch.zeros(3))

        _init_head_zero(self.head_pose)

        if self.additive_gamma_correction:
            _init_head_zero(self.head_gamma)
        else:
            _init_linear_small(self.head_gamma, std=1e-3)
            if self.gamma_log_span is None:
                target_sigmoid = (1.0 - self.gamma_min) / (self.gamma_max - self.gamma_min)
                target_sigmoid = max(1e-6, min(1.0 - 1e-6, target_sigmoid))
                bias_gamma_val = math.log(target_sigmoid / (1.0 - target_sigmoid))
                nn.init.constant_(self.head_gamma.bias, bias_gamma_val)
            else:
                nn.init.zeros_(self.head_gamma.bias)

    def _expr_landmarks(self, mp_landmarks_2d, mp_landmarks_3d, world_to_cam, device, dtype):
        if mp_landmarks_3d is not None and world_to_cam is not None:
            lmk = landmarks_3d_to_camera_xy(
                mp_landmarks_3d.to(device=device, dtype=dtype), world_to_cam
            )
            return lmk.reshape(lmk.shape[0], -1)
        if mp_landmarks_2d is None:
            raise ValueError(
                "expr trunk with use_landmarks requires mp_landmarks_2d or mp_landmarks_3d+world_to_cam"
            )
        lmk = mp_landmarks_2d
        if self.canonicalize_landmarks_2d:
            lmk = landmarks_2d_canonical(lmk)
        return lmk.reshape(lmk.shape[0], -1)

    def _mp_rotation_6d(self, mp_pose_raw, mp_transform_matrix, B, device, dtype, camera_R=None):
        """MediaPipe head rotation as Zhou 6D (detector base, no grad)."""
        if mp_transform_matrix is not None:
            T = mp_transform_matrix
            if not isinstance(T, torch.Tensor):
                T = torch.as_tensor(T, device=device, dtype=dtype)
            else:
                T = T.to(device=device, dtype=dtype)
            if T.ndim == 2:
                T = T.unsqueeze(0)
            rows = []
            for i in range(T.shape[0]):
                R = T[i, :3, :3]
                s = R.norm(dim=0).mean().clamp(min=1e-8)
                R_norm = R / s
                if camera_R is not None:
                    R_cam = camera_R.to(device=device, dtype=dtype)
                    if R_cam.ndim == 2:
                        R_norm = R_cam.T @ R_norm @ R_cam
                    else:
                        R_norm = R_cam.transpose(-1, -2) @ R_norm @ R_cam
                rows.append(rotation_matrix_to_6d(R_norm))
            mp_r6 = torch.stack(rows, dim=0)
        else:
            mp_r6 = mp_pose_raw[..., :6].to(device=device, dtype=dtype)
            if camera_R is not None:
                R_mp = rotation_6d_to_matrix(mp_r6)
                R_cam = camera_R.to(device=device, dtype=dtype)
                if R_cam.ndim == 2:
                    R_cam = R_cam.unsqueeze(0)
                R_world = R_cam.transpose(-1, -2) @ R_mp @ R_cam
                mp_r6 = rotation_matrix_to_6d(R_world)
            near_zero = mp_r6.abs().amax(dim=-1) < 1e-6
            if near_zero.any():
                mp_r6 = torch.where(
                    near_zero.unsqueeze(-1),
                    self.identity_6d.to(device=device, dtype=dtype).expand_as(mp_r6),
                    mp_r6,
                )
        return mp_r6.detach()

    def _ict_expression_weights(
        self,
        ict_raw: torch.Tensor,
        raw_gamma: torch.Tensor,
        *,
        force_gamma_one: bool,
        additive_gamma_correction: bool,
    ):
        if additive_gamma_correction:
            gamma_delta = raw_gamma
            if force_gamma_one:
                gamma_delta = torch.zeros_like(raw_gamma)
            weights = ict_raw + gamma_delta
            return weights, gamma_delta, True

        if self.gamma_log_span is not None:
            t = 2.0 * torch.sigmoid(raw_gamma) - 1.0
            gamma = torch.exp(self.gamma_log_span * t)
        else:
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(raw_gamma)
        if force_gamma_one:
            gamma = torch.ones_like(gamma)
        weights = ict_raw.pow(gamma)
        return weights, gamma, False

    def forward(
        self,
        mp_blendshape,
        mp_landmarks_2d=None,
        mp_landmarks_3d=None,
        world_to_cam=None,
        mp_pose_raw=None,
        mp_transform_matrix=None,
        force_gamma_one=False,
        camera_R=None,
        use_global_translation_param=False,
        additive_gamma_correction=None,
    ):
        B = mp_blendshape.shape[0]
        device = mp_blendshape.device
        dtype = mp_blendshape.dtype

        if mp_blendshape.shape[-1] != self.n_blendshapes:
            raise ValueError(
                f"mp_blendshape last dim {mp_blendshape.shape[-1]} != n_blendshapes {self.n_blendshapes}"
            )

        if mp_pose_raw is None:
            mp_pose_raw = torch.zeros(B, self.pose_dim, device=device, dtype=dtype)
        elif not isinstance(mp_pose_raw, torch.Tensor):
            mp_pose_raw = torch.as_tensor(mp_pose_raw, device=device, dtype=dtype)
        if mp_pose_raw.ndim == 1:
            mp_pose_raw = mp_pose_raw.unsqueeze(0)
        if mp_pose_raw.shape[-1] == 16:
            mp_pose_raw = mp_pose_raw[..., :6]

        c_raw = mp_blendshape.clamp(1e-6, 1.0)

        ict_raw = c_raw[:, self.mediapipe_to_ict]

        expr_parts = [mp_blendshape]
        if self.use_landmarks:
            expr_parts.append(
                self._expr_landmarks(mp_landmarks_2d, mp_landmarks_3d, world_to_cam, device, dtype)
            )
        h_expr = self.expr_trunk(torch.cat(expr_parts, dim=-1))
        raw_gamma = self.head_gamma(h_expr)
        use_additive = (
            self.additive_gamma_correction
            if additive_gamma_correction is None
            else bool(additive_gamma_correction)
        )
        ict_expression_weights, gamma, is_additive = self._ict_expression_weights(
            ict_raw,
            raw_gamma,
            force_gamma_one=force_gamma_one,
            additive_gamma_correction=use_additive,
        )
        coeffs_corr = ict_expression_weights

        mp_r6 = self._mp_rotation_6d(mp_pose_raw, mp_transform_matrix, B, device, dtype, camera_R=camera_R)

        pose_lmk = gather_mp_landmarks(mp_landmarks_2d, POSE_LMK_MP, device, dtype)
        h_pose = self.pose_trunk(torch.cat([mp_pose_raw, pose_lmk], dim=-1))
        pose_delta = self.head_pose(h_pose)
        pose_residual = mp_r6 + pose_delta[:, :6]
        translation_residual = pose_delta[:, 6:9]
        if use_global_translation_param:
            translation_global = self.global_translation.to(device=device, dtype=dtype).unsqueeze(0).expand(
                B, -1
            )
        else:
            translation_global = pose_delta[:, 9:12]
        scale = torch.exp(self.log_pose_scale).to(device=device, dtype=dtype).expand(B)

        return {
            "gamma": gamma,
            "additive_gamma": is_additive,
            "coeffs": coeffs_corr,
            "coeffs_raw": ict_raw,
            "ict_expression_weights": ict_expression_weights,
            "pose_scale": scale,
            "pose_residual": pose_residual,
            "mp_rotation_6d": mp_r6,
            "pose_rotation_delta": pose_delta[:, :6],
            "translation_residual": translation_residual,
            "translation_global": translation_global,
        }
