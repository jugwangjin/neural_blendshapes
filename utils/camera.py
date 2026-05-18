"""Fixed camera for 3DGS + MediaPipe projection (no FLARE Camera class)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from utils.default_camera import DEFAULT_CAMERA_NPZ, load_default_camera


@dataclass
class FixedCamera:
    width: int = 512
    height: int = 512
    fx: float = 1508.0
    fy: float = 1508.0
    cx: float = 256.0
    cy: float = 256.0
    R: torch.Tensor = None
    t: torch.Tensor = None

    def __post_init__(self):
        if self.R is None:
            self.R = torch.eye(3)
        if self.t is None:
            self.t = torch.zeros(3)

    @property
    def K(self):
        k = torch.eye(3, dtype=torch.float32)
        k[0, 0] = self.fx
        k[1, 1] = self.fy
        k[0, 2] = self.cx
        k[1, 2] = self.cy
        return k

    @classmethod
    def from_default_npz(cls, path=None, width=512, height=512):
        d = load_default_camera(path or DEFAULT_CAMERA_NPZ)
        k = d["K_mean"]
        return cls(
            width=width,
            height=height,
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            R=torch.tensor(d["R_mean"], dtype=torch.float32),
            t=torch.tensor(d["t_mean"], dtype=torch.float32),
        )


def world_to_camera(points, cam: FixedCamera):
    """points [..., 3] in world; returns camera-space points."""
    R = cam.R.to(points.device, dtype=points.dtype)
    t = cam.t.to(points.device, dtype=points.dtype)
    return points @ R.T + t


def project_points(points_cam, cam: FixedCamera):
    """Perspective project camera-space points to pixel coords [..., 2]."""
    z = points_cam[..., 2:3].clamp(min=1e-6)
    x = points_cam[..., 0:1] / z
    y = points_cam[..., 1:2] / z
    u = cam.fx * x + cam.cx
    v = cam.fy * y + cam.cy
    return torch.cat([u, v], dim=-1).squeeze(-1) if u.ndim > 1 else torch.stack([u, v], dim=-1)


def project_world_points(points, cam: FixedCamera):
    return project_points(world_to_camera(points, cam), cam)
