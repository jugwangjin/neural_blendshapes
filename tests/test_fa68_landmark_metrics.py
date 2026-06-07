"""Unit tests for FA-68 mesh vs face_alignment landmark error metrics."""

import torch

from eval.fa68_landmark_metrics import Fa68ErrorAccumulator, fa68_landmark_error_batch
from utils.camera import FixedCamera


class _FakeIct:
    def __init__(self, n68: int = 68):
        self.landmark_indices = list(range(n68))

    def landmark_vertices(self, mesh, region="all"):
        idx = torch.tensor(self.landmark_indices, device=mesh.device, dtype=torch.long)
        return mesh[:, idx]


def _identity_camera(image_size: int = 512, device="cpu"):
    return FixedCamera(
        width=image_size,
        height=image_size,
        fx=512.0,
        fy=512.0,
        cx=256.0,
        cy=256.0,
        R=torch.eye(3, device=device),
        t=torch.tensor([0.0, 0.0, 5.0], device=device),
    )


def test_fa68_landmark_error_zero_when_aligned():
    device = torch.device("cpu")
    image_size = 512
    cam = _identity_camera(image_size, device)
    ict = _FakeIct()
    vertices = torch.zeros(1, 68, 3, device=device)
    vertices[0, :, 2] = 5.0
    proj = cam.project_world_points(vertices.reshape(-1, 3)).reshape(1, 68, 2)
    fa = torch.zeros(1, 68, 4, device=device)
    fa[:, :, :2] = proj / float(image_size)
    fa[:, :, 3] = 1.0
    stats = fa68_landmark_error_batch(vertices, ict, fa, cam, image_size)[0]
    assert stats["n_valid"] == 68
    assert stats["mse_px"] == 0.0


def test_fa68_accumulator_mse_and_std():
    acc = Fa68ErrorAccumulator()
    acc.add_frame({"n_valid": 2, "mse_px": 4.0, "rmse_px": 2.0, "per_point_rmse_px": [2.0, 2.0]})
    acc.add_frame({"n_valid": 2, "mse_px": 16.0, "rmse_px": 4.0, "per_point_rmse_px": [4.0, 4.0]})
    s = acc.summary()
    assert abs(s["mse_px"] - 10.0) < 1e-4
    assert abs(s["per_frame_mse_std"] - 6.0) < 1e-4
