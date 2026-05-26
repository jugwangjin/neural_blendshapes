"""Fixed camera for 3DGS + MediaPipe projection (ICT / training stack)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_NPZ = REPO_ROOT / "assets" / "default_camera.npz"
DEFAULT_CAMERA_TXT = REPO_ROOT / "assets" / "default_camera.txt"


def load_default_camera(path=None):
    path = Path(path or DEFAULT_CAMERA_NPZ)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def save_default_camera(
    R,
    t,
    K,
    path=None,
    *,
    train_view_corrected: bool = False,
    image_size: int | None = None,
    bbox_scale: float | None = None,
):
    path = Path(path or DEFAULT_CAMERA_NPZ)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "K_mean": np.asarray(K, dtype=np.float64),
        "R_mean": np.asarray(R, dtype=np.float64),
        "t_mean": np.asarray(t, dtype=np.float64).reshape(3),
        "train_view_corrected": np.array(bool(train_view_corrected)),
    }
    if image_size is not None:
        payload["image_size"] = np.array(int(image_size))
    if bbox_scale is not None:
        payload["bbox_scale"] = np.array(float(bbox_scale))
    np.savez(str(path), **payload)
    return path


def _train_view_baked(meta: dict | None) -> bool:
    if meta is None or "train_view_corrected" not in meta:
        return False
    return bool(np.asarray(meta["train_view_corrected"]).item())


def load_training_camera(verts, *, path, width: int, height: int, device=None):
    path = Path(path)
    meta = load_default_camera(path) if path.is_file() else None
    cam = FixedCamera.from_default_or_mesh(
        verts,
        path=path,
        width=width,
        height=height,
        device=device,
    )
    if meta is not None and _train_view_baked(meta):
        return cam
    pivot = verts.detach().float().reshape(-1, 3).mean(dim=0)
    return cam.with_view_correction(pivot)


def training_camera_status(path) -> str:
    path = Path(path)
    if not path.is_file():
        return f"missing ({path}) — mesh-bounds + runtime view correction"
    meta = load_default_camera(path)
    if _train_view_baked(meta):
        return f"{path} (train view baked in R_mean)"
    return f"{path} (legacy — runtime view correction applied)"


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
    def from_default_npz(cls, path=None, width=512, height=512, device=None):
        d = load_default_camera(path or DEFAULT_CAMERA_NPZ)
        if d is None:
            raise FileNotFoundError(
                f"default camera not found: {path or DEFAULT_CAMERA_NPZ}. "
                "Run processing/compute_camera_for_metrical_crop.py --apply-train-view --write-npz"
            )
        k = d["K_mean"]
        dev = torch.device("cpu") if device is None else torch.device(device)
        cam = cls(
            width=width,
            height=height,
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            R=torch.tensor(d["R_mean"], dtype=torch.float32, device=dev),
            t=torch.tensor(d["t_mean"], dtype=torch.float32, device=dev),
        )
        return cam

    @classmethod
    def from_mesh_bounds(
        cls,
        verts,
        width=512,
        height=512,
        fov_deg=35.0,
        margin=1.35,
        axis=2,
    ):
        """
        Pinhole camera framing ``verts`` (OpenCV: +Z forward in camera space).

        Row-vector convention: ``p_cam = p_world @ R.T + t`` with ``R=I``,
        ``t = -center`` plus offset along ``axis`` so the mesh centroid has positive depth.
        """
        v = verts.detach().float().reshape(-1, 3)
        dev = v.device
        center = v.mean(dim=0)
        extent = (v.max(dim=0).values - v.min(dim=0).values).max().item()
        radius = max(extent * margin * 0.5, 0.05)
        dist = radius / np.tan(np.radians(fov_deg / 2.0)) * 1.1
        fx = fy = (width * 0.5) / np.tan(np.radians(fov_deg / 2.0))
        t = -center.clone()
        t[axis] = t[axis] + dist
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=width * 0.5,
            cy=height * 0.5,
            R=torch.eye(3, dtype=torch.float32, device=dev),
            t=t.to(dtype=torch.float32),
        )

    @classmethod
    def from_default_or_mesh(cls, verts, path=None, width=512, height=512, device=None, **mesh_kw):
        """Load baked npz if present; otherwise fit to ``verts``."""
        dev = device
        if dev is None and torch.is_tensor(verts):
            dev = verts.device
        d = load_default_camera(path or DEFAULT_CAMERA_NPZ)
        if d is not None:
            return cls.from_default_npz(path=path, width=width, height=height, device=dev)
        return cls.from_mesh_bounds(verts, width=width, height=height, **mesh_kw)

    def to(self, device):
        device = torch.device(device)
        return FixedCamera(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            R=self.R.to(device),
            t=self.t.to(device),
        )

    def with_azimuth_y(self, azimuth_deg, pivot):
        """
        Orbit camera around world +Y (azimuth, degrees).

        +azimuth = camera moves to subject's right (see face from the left).
        Row convention: ``p_cam = p @ R.T + t``.
        """
        device = self.R.device
        dtype = self.R.dtype
        cam = self.to(device)
        pivot = pivot.detach().float().reshape(3).to(device=device, dtype=dtype)
        R_delta = rotation_matrix_y_deg(-float(azimuth_deg), device=device, dtype=dtype)
        cam.R = R_delta @ cam.R
        cam.t = (pivot - pivot @ R_delta.T) @ cam.R.T + cam.t
        return cam

    def with_roll_forward_deg(self, roll_deg):
        """
        Roll about camera forward (+Z in OpenCV camera space).

        180° fixes upside-down image when ``from_mesh_bounds`` uses ``R=I`` on FLAME-aligned ICT.
        """
        device = self.R.device
        dtype = self.R.dtype
        cam = self.to(device)
        R_roll = rotation_matrix_z_deg(float(roll_deg), device=device, dtype=dtype)
        cam.R = R_roll @ cam.R
        cam.t = cam.t @ R_roll.T
        return cam

    def with_view_correction(self, pivot, yaw_deg=180.0, roll_deg=180.0):
        """
        Frontal sanity/train view for FLAME-aligned ICT (face toward -Z, mesh bounds on +Z).

        Order: orbit ``yaw_deg`` about world +Y at ``pivot``, then ``roll_deg`` about camera +Z.
        """
        device = self.R.device
        cam = self.to(device)
        if yaw_deg != 0.0:
            cam = cam.with_azimuth_y(yaw_deg, pivot)
        if roll_deg != 0.0:
            cam = cam.with_roll_forward_deg(roll_deg)
        return cam


def rotation_matrix_y_deg(angle_deg, device="cpu", dtype=torch.float32):
    """Right-handed rotation about world +Y (azimuth), for row-vector points ``p @ R.T``."""
    rad = float(angle_deg) * np.pi / 180.0
    c = torch.tensor(np.cos(rad), device=device, dtype=dtype)
    s = torch.tensor(np.sin(rad), device=device, dtype=dtype)
    z = torch.zeros((), device=device, dtype=dtype)
    o = torch.ones((), device=device, dtype=dtype)
    return torch.stack(
        [
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ]
    )


def rotation_matrix_z_deg(angle_deg, device="cpu", dtype=torch.float32):
    """Right-handed rotation about +Z (roll in OpenCV camera frame)."""
    rad = float(angle_deg) * np.pi / 180.0
    c = torch.tensor(np.cos(rad), device=device, dtype=dtype)
    s = torch.tensor(np.sin(rad), device=device, dtype=dtype)
    z = torch.zeros((), device=device, dtype=dtype)
    o = torch.ones((), device=device, dtype=dtype)
    return torch.stack(
        [
            torch.stack([c, -s, z]),
            torch.stack([s, c, z]),
            torch.stack([z, z, o]),
        ]
    )


def rotate_points_y(points, angle_deg, pivot=None):
    """Rotate ``[..., 3]`` about world +Y through ``pivot`` (default: centroid)."""
    pts = points.detach().float()
    flat = pts.reshape(-1, 3)
    if pivot is None:
        pivot = flat.mean(dim=0)
    else:
        pivot = pivot.detach().float().reshape(3)
    R = rotation_matrix_y_deg(angle_deg, device=flat.device, dtype=flat.dtype)
    out = (flat - pivot) @ R.T + pivot
    return out.reshape(pts.shape)


def default_azimuth_sweep(step_deg=30.0):
    """Integer azimuth list from -90 to +90 inclusive, step 30°."""
    step = int(step_deg)
    return list(range(-90, 90 + 1, step))


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


FixedCamera.project_world_points = lambda self, points: project_world_points(points, self)
