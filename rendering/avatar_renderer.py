"""
Avatar renderer — gsplat.rasterization only.

Requires: pip install gsplat  OR  pip install -e ./gsplat (submodule, do not edit).
"""

import torch
import torch.nn as nn
from gsplat import rasterization

from rendering.gsplat_camera import fixed_camera_to_gsplat
from rendering.pack import pack_gaussians


class AvatarRenderer(nn.Module):
    def __init__(
        self,
        cfg=None,
        image_size=512,
        znear=0.01,
        zfar=100.0,
        bg_color=None,
        channel_chunk=32,
        sh_degree=None,
    ):
        super().__init__()
        self.image_size = image_size
        self.znear = znear
        self.zfar = zfar
        self.channel_chunk = channel_chunk
        if sh_degree is None and cfg is not None:
            sh_degree = getattr(cfg, "sh_degree", None)
        self.sh_degree = sh_degree

        if bg_color is None:
            bg = torch.zeros(3)
        else:
            bg = torch.tensor(bg_color, dtype=torch.float32)
        self.register_buffer("background_rgb", bg)

    def _cam(self, camera):
        return fixed_camera_to_gsplat(camera, znear=self.znear, zfar=self.zfar)

    def set_sh_degree(self, sh_degree):
        self.sh_degree = sh_degree

    def _rasterize(self, packed, camera, features, render_mode="RGB", backgrounds=None, sh_degree=None):
        cam = self._cam(camera)
        backgrounds = backgrounds.to(device=packed["means"].device, dtype=features.dtype)
        if sh_degree is None:
            sh_degree = self.sh_degree
        return rasterization(
            means=packed["means"],
            quats=packed["quats"],
            scales=packed["scales"],
            opacities=packed["opacities"].reshape(-1),
            colors=features,
            viewmats=cam["viewmats"],
            Ks=cam["Ks"],
            width=cam["width"],
            height=cam["height"],
            near_plane=cam["znear"],
            far_plane=cam["zfar"],
            sh_degree=sh_degree,
            render_mode=render_mode,
            backgrounds=backgrounds,
            channel_chunk=self.channel_chunk,
            packed=True,
        )

    @staticmethod
    def _to_nchw(render_colors, render_alphas):
        return render_colors.permute(0, 3, 1, 2).contiguous(), render_alphas.permute(0, 3, 1, 2).contiguous()

    def render_rgb(self, avatar_out, camera, background=None):
        packed = pack_gaussians(avatar_out)
        bg = self.background_rgb.to(packed["means"].device)
        if background is not None:
            bg = background.to(packed["means"].device).reshape(-1)[:3]
        colors, alphas, meta = self._rasterize(
            packed, camera, packed["colors"], backgrounds=bg.view(1, 3)
        )
        rgb, alpha = self._to_nchw(colors, alphas)
        return {
            "rgb": rgb.clamp(0, 1),
            "alpha": alpha,
            "meta": meta,
            "viewspace_points": meta.get("means2d"),
            "radii": meta.get("radii"),
        }

    def render_depth(self, avatar_out, camera, render_mode="ED"):
        packed = pack_gaussians(avatar_out)
        device = packed["means"].device
        colors, alphas, meta = self._rasterize(
            packed,
            camera,
            packed["colors"],
            render_mode=render_mode,
            backgrounds=torch.zeros(1, 1, device=device),
        )
        depth, alpha = self._to_nchw(colors, alphas)
        return {"depth": depth, "alpha": alpha, "meta": meta}

    def render_features(self, avatar_out, camera, features=None, backgrounds=None):
        packed = pack_gaussians(avatar_out, rgb_activation=None)
        features = features if features is not None else packed["sem_prob"]
        K = features.shape[-1]
        device = features.device
        if backgrounds is None:
            backgrounds = torch.zeros(1, K, device=device, dtype=features.dtype)
        colors, alphas, meta = self._rasterize(packed, camera, features, backgrounds=backgrounds)
        sem, alpha = self._to_nchw(colors, alphas)
        return {
            "semantic": sem,
            "semantic_prob": sem / alpha.clamp(min=1e-6),
            "alpha": alpha,
            "meta": meta,
        }

    def forward(self, avatar_out, camera, render_semantic=True, background=None):
        out = self.render_rgb(avatar_out, camera, background=background)
        if render_semantic and avatar_out.get("sem_prob") is not None:
            sem_out = self.render_features(avatar_out, camera)
            out["semantic"] = sem_out["semantic"]
            out["semantic_prob"] = sem_out["semantic_prob"]
            out["semantic_alpha"] = sem_out["alpha"]
        return out
