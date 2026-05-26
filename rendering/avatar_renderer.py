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
        self.packed = bool(getattr(cfg, "gsplat_packed", False) if cfg is not None else False)

        if bg_color is None:
            bg = torch.zeros(3)
        else:
            bg = torch.tensor(bg_color, dtype=torch.float32)
        self.register_buffer("background_rgb", bg)

    def _cam(self, camera, device):
        return fixed_camera_to_gsplat(
            camera, znear=self.znear, zfar=self.zfar, device=device
        )

    def set_sh_degree(self, sh_degree):
        self.sh_degree = sh_degree

    @staticmethod
    def _bg_nchw(background, channels, device, dtype):
        bg = background.to(device=device, dtype=dtype).reshape(-1)
        if bg.numel() == 1:
            bg = bg.expand(channels)
        else:
            bg = bg[:channels]
        if bg.numel() < channels:
            bg = torch.cat([bg, bg.new_zeros(channels - bg.numel(), device=device, dtype=dtype)])
        return bg.view(1, channels, 1, 1)

    @staticmethod
    def _composite_background(image_nchw, alpha_nchw, background_nchw):
        return image_nchw + (1.0 - alpha_nchw).clamp(0.0, 1.0) * background_nchw

    def _rasterize(self, packed, camera, features, render_mode="RGB", sh_degree=None):
        device = packed["means"].device
        cam = self._cam(camera, device=device)
        if sh_degree is None:
            sh_degree = self.sh_degree
        # Do not pass backgrounds into gsplat (packed=True shape assert in rasterize_to_pixels).
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
            backgrounds=None,
            channel_chunk=self.channel_chunk,
            packed=self.packed,
        )

    @staticmethod
    def _to_nchw(render_colors, render_alphas):
        return render_colors.permute(0, 3, 1, 2).contiguous(), render_alphas.permute(0, 3, 1, 2).contiguous()

    def render_rgb(self, avatar_out, camera, background=None):
        packed = pack_gaussians(avatar_out)
        device = packed["means"].device
        bg = self.background_rgb.to(device)
        if background is not None:
            bg = background.to(device).reshape(-1)[:3]
        colors, alphas, meta = self._rasterize(packed, camera, packed["colors"])
        rgb, alpha = self._to_nchw(colors, alphas)
        rgb = self._composite_background(rgb, alpha, self._bg_nchw(bg, 3, device, rgb.dtype))
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
            packed, camera, packed["colors"], render_mode=render_mode
        )
        depth, alpha = self._to_nchw(colors, alphas)
        return {"depth": depth, "alpha": alpha, "meta": meta}

    def render_features(self, avatar_out, camera, features=None, backgrounds=None):
        packed = pack_gaussians(avatar_out, rgb_activation=None)
        features = features if features is not None else packed["sem_prob"]
        K = features.shape[-1]
        device = features.device
        if backgrounds is None:
            backgrounds = torch.zeros(K, device=device, dtype=features.dtype)
        colors, alphas, meta = self._rasterize(packed, camera, features)
        sem, alpha = self._to_nchw(colors, alphas)
        sem = self._composite_background(sem, alpha, self._bg_nchw(backgrounds, K, device, sem.dtype))
        return {
            "semantic": sem,
            "semantic_prob": sem / alpha.clamp(min=1e-6),
            "alpha": alpha,
            "meta": meta,
        }

    def render_expected_signal(self, avatar_out, camera, signal, signal_dim=1):
        """
        Alpha-weighted signal render (depth ``ED``-style, **no background composite**).

        ``signal``: ``[G]`` or ``[G, C]`` per-Gaussian values (e.g. ``h``).
        Returns ``accum`` = Σ w_i s_i, ``expected`` = accum / alpha, ``alpha`` = Σ w_i.
        Use ``accum`` with pixel mask ``alpha * gt_mask`` to avoid low-alpha blow-up;
        do **not** use RGB-style ``accum + (1-alpha)*bg``.
        """
        packed = pack_gaussians(avatar_out, rgb_activation=None)
        sig = signal.reshape(-1, signal.shape[-1] if signal.ndim > 1 else 1).float()
        if sig.shape[-1] < signal_dim:
            pad = sig.new_zeros(sig.shape[0], signal_dim - sig.shape[-1])
            sig = torch.cat([sig, pad], dim=-1)
        sig = sig[:, :signal_dim]
        if sig.shape[-1] == 1:
            colors = sig.expand(-1, 3)
        else:
            colors = sig
            if colors.shape[-1] == 2:
                colors = torch.cat([colors, colors.new_zeros(colors.shape[0], 1)], dim=-1)
        colors, alphas, meta = self._rasterize(packed, camera, colors)
        accum, alpha = self._to_nchw(colors, alphas)
        accum = accum[:, :signal_dim]
        expected = accum / alpha.clamp(min=1e-6)
        return {"accum": accum, "expected": expected, "alpha": alpha, "meta": meta}

    def forward(self, avatar_out, camera, render_semantic=True, background=None):
        out = self.render_rgb(avatar_out, camera, background=background)
        if render_semantic and avatar_out.get("sem_prob") is not None:
            sem_out = self.render_features(avatar_out, camera)
            out["semantic"] = sem_out["semantic"]
            out["semantic_prob"] = sem_out["semantic_prob"]
            out["semantic_alpha"] = sem_out["alpha"]
        return out
