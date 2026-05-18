"""MediaPipe iris landmarks (468–477) vs iris control Gaussians."""

import torch


def loss_iris_landmarks_2d(iris_xyz, mp_uv, camera, image_size):
    """
    iris_xyz: [10, 3] world (5 left + 5 right control points)
    mp_uv: [478, 2] normalized [0,1] MediaPipe landmarks
    """
    left_idx = [468, 469, 470, 471, 472]
    right_idx = [473, 474, 475, 476, 477]
    targets = torch.cat([mp_uv[left_idx], mp_uv[right_idx]], dim=0)

    proj = camera.project_world_points(iris_xyz)
    pred = proj[:, :2] / image_size
    return torch.nn.functional.mse_loss(pred, targets)
