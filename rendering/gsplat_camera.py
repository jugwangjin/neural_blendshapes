"""FixedCamera → gsplat viewmats + intrinsics (OpenCV pinhole)."""

import torch

from utils.camera import FixedCamera


def fixed_camera_to_gsplat(cam: FixedCamera, znear=0.01, zfar=100.0, device=None):
    device = device if device is not None else cam.R.device
    dtype = cam.R.dtype
    w2v = torch.eye(4, device=device, dtype=dtype)
    w2v[:3, :3] = cam.R.to(device=device, dtype=dtype)
    w2v[:3, 3] = cam.t.to(device=device, dtype=dtype)
    viewmats = w2v.unsqueeze(0)
    K = cam.K.to(device=device, dtype=dtype).unsqueeze(0)
    return {
        "viewmats": viewmats,
        "Ks": K,
        "width": int(cam.width),
        "height": int(cam.height),
        "znear": znear,
        "zfar": zfar,
    }
