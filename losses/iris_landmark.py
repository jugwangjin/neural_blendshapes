"""MediaPipe iris landmarks (468–477) vs iris control Gaussians."""

import torch

from utils.mediapipe_indices import LEFT_IRIS_MP, RIGHT_IRIS_MP


def loss_iris_landmarks_2d(iris_xyz, mp_uv, camera, image_size):
    """
    iris_xyz: [10, 3] world (5 left + 5 right control points)
    mp_uv: [478, 2] normalized [0,1] MediaPipe landmarks
    """
    targets = torch.cat([mp_uv[LEFT_IRIS_MP], mp_uv[RIGHT_IRIS_MP]], dim=0)

    proj = camera.project_world_points(iris_xyz)
    pred = proj[:, :2] / image_size
    return torch.nn.functional.mse_loss(pred, targets)
