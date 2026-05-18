import torch

from model.pose_weight import PoseWeightMLP


def test_pose_weight_range():
    net = PoseWeightMLP()
    xyz = torch.randn(100, 3)
    w = net(xyz)
    assert w.shape == (100, 1)
    assert w.min() >= 0 and w.max() <= 1
