"""Face-alignment 68 vs ICT mesh landmark error (independent of MediaPipe training)."""

from __future__ import annotations

import torch

from utils.camera import world_to_camera


def fa68_mesh_pred_uv(vertices, ict, camera, image_size: int):
    """Project ICT Multi-PIE 68 vertex landmarks to normalized UV [0, 1]."""
    lmk_xyz = ict.landmark_vertices(vertices, region="all")
    proj = camera.project_world_points(lmk_xyz.reshape(-1, 3)).reshape(
        lmk_xyz.shape[0], lmk_xyz.shape[1], 2
    )
    return proj / float(image_size), lmk_xyz


@torch.no_grad()
def fa68_landmark_error_batch(
    vertices,
    ict,
    landmark_fa,
    camera,
    image_size: int,
    *,
    score_thresh: float = 0.3,
):
    """Compare mesh-indexed 68 landmarks vs face_alignment detections."""
    pred_uv, lmk_xyz = fa68_mesh_pred_uv(vertices, ict, camera, image_size)
    b, n68, _ = pred_uv.shape
    fa = landmark_fa.to(device=vertices.device, dtype=vertices.dtype)
    target_uv = fa[:, :n68, :2]
    scores = fa[:, :n68, 3]

    lmk_cam = world_to_camera(lmk_xyz, camera)
    in_front = lmk_cam[..., 2] > 1e-3
    valid = (scores >= float(score_thresh)) & in_front

    diff_px = (pred_uv - target_uv) * float(image_size)
    sq_dist_px = (diff_px**2).sum(dim=-1)

    out = []
    for i in range(b):
        m = valid[i]
        n_valid = int(m.sum().item())
        if n_valid == 0:
            out.append(
                {
                    "n_valid": 0,
                    "mse_px": float("nan"),
                    "rmse_px": float("nan"),
                    "mae_px": float("nan"),
                    "per_point_rmse_px": [],
                }
            )
            continue
        sq = sq_dist_px[i, m]
        dist = torch.sqrt(sq)
        mse = float(sq.mean().item())
        out.append(
            {
                "n_valid": n_valid,
                "mse_px": mse,
                "rmse_px": float(torch.sqrt(sq.mean()).item()),
                "mae_px": float(dist.mean().item()),
                "per_point_rmse_px": dist.detach().cpu().tolist(),
            }
        )
    return out


class Fa68ErrorAccumulator:
    """Aggregate frame-level FA68 errors into run-level MSE / std."""

    def __init__(self):
        self._frame_mse: list[float] = []
        self._frame_rmse: list[float] = []
        self._all_sq_px: list[float] = []
        self._n_frames = 0
        self._n_skipped = 0

    def add_frame(self, frame_stats: dict):
        self._n_frames += 1
        if frame_stats["n_valid"] == 0:
            self._n_skipped += 1
            return
        self._frame_mse.append(frame_stats["mse_px"])
        self._frame_rmse.append(frame_stats["rmse_px"])
        for d in frame_stats["per_point_rmse_px"]:
            self._all_sq_px.append(float(d) ** 2)

    def summary(self) -> dict:
        if not self._frame_mse:
            return {
                "n_frames": self._n_frames,
                "n_frames_valid": 0,
                "n_frames_skipped": self._n_skipped,
                "n_landmarks_valid": 0,
                "mse_px": float("nan"),
                "rmse_px": float("nan"),
                "per_frame_mse_mean": float("nan"),
                "per_frame_mse_std": float("nan"),
                "per_frame_rmse_mean": float("nan"),
                "per_frame_rmse_std": float("nan"),
            }

        frame_mse = torch.tensor(self._frame_mse, dtype=torch.float64)
        frame_rmse = torch.tensor(self._frame_rmse, dtype=torch.float64)
        all_sq = torch.tensor(self._all_sq_px, dtype=torch.float64)
        return {
            "n_frames": self._n_frames,
            "n_frames_valid": len(self._frame_mse),
            "n_frames_skipped": self._n_skipped,
            "n_landmarks_valid": len(self._all_sq_px),
            "mse_px": float(all_sq.mean().item()),
            "rmse_px": float(torch.sqrt(all_sq.mean()).item()),
            "per_frame_mse_mean": float(frame_mse.mean().item()),
            "per_frame_mse_std": float(frame_mse.std(unbiased=False).item())
            if frame_mse.numel() > 1
            else 0.0,
            "per_frame_rmse_mean": float(frame_rmse.mean().item()),
            "per_frame_rmse_std": float(frame_rmse.std(unbiased=False).item())
            if frame_rmse.numel() > 1
            else 0.0,
        }
