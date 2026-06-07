"""
Avatar renderer — gsplat.rasterization only.

Requires: pip install gsplat  OR  pip install -e ./gsplat (submodule, do not edit).
"""

import torch
import torch.nn as nn
from gsplat import rasterization

from rendering.gsplat_camera import fixed_camera_to_gsplat
from rendering.pack import pack_gaussians, pack_gaussians_silhouette


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
        # "classic" ≈ original 3DGS / GB diff_gaussian_rasterization (no mip antialiasing).
        # "antialiased" = gsplat Mip-Splatting-style compensation — off by default.
        self.rasterize_mode = str(getattr(cfg, "gsplat_rasterize_mode", "classic"))

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
            # [N, K, 3] = SH coeffs; [N, C] = flat RGB / semantic / signal (not SH).
            sh_degree = self.sh_degree if features.ndim == 3 else None

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
            rasterize_mode=self.rasterize_mode,
            backgrounds=None,
            channel_chunk=self.channel_chunk,
            packed=self.packed,
        )

    @staticmethod
    def _to_nchw(render_colors, render_alphas):
        return render_colors.permute(0, 3, 1, 2).contiguous(), render_alphas.permute(0, 3, 1, 2).contiguous()

    def render_silhouette_alpha(self, avatar_out, camera, *, detach_covariance=True):
        """
        Legacy / debug: detached scale+rotation silhouette pass.

        Training uses ``render_rgb`` alpha on one gsplat forward (GB ``render_alpha``).
        """
        packed = pack_gaussians_silhouette(
            avatar_out,
            rgb_activation=None,
            detach_covariance=detach_covariance,
        )
        colors, alphas, meta = self._rasterize(packed, camera, packed["colors"])
        _, alpha = self._to_nchw(colors, alphas)
        return {
            "alpha": alpha,
            "meta": meta,
            "viewspace_points": meta.get("means2d"),
            "radii": meta.get("radii"),
        }

    def render_rgb(self, avatar_out, camera, background=None, *, composite=True):
        packed = pack_gaussians(avatar_out, sh_degree=self.sh_degree)
        device = packed["means"].device
        bg = self.background_rgb.to(device)
        if background is not None:
            bg = background.to(device).reshape(-1)[:3]
        colors, alphas, meta = self._rasterize(packed, camera, packed["colors"])
        rgb, alpha = self._to_nchw(colors, alphas)
        if composite:
            rgb = self._composite_background(rgb, alpha, self._bg_nchw(bg, 3, device, rgb.dtype))
        return {
            "rgb": rgb.clamp(0, 1),
            "alpha": alpha,
            "meta": meta,
            "viewspace_points": meta.get("means2d"),
            "radii": meta.get("radii"),
        }

    def render_rgb_and_semantic(self, avatar_out, camera, background=None):
        """
        Single gsplat pass: RGB (sigmoid color, scene bg) + semantic one-hot (bg=0).

        Use when training needs both ``rgb`` and ``semantic``; inference / RGB-only
        should call ``render_rgb`` or ``forward(..., render_semantic=False)``.
        """
        packed = pack_gaussians(avatar_out, sh_degree=self.sh_degree)
        sem = packed.get("sem_features")
        if sem is None:
            return self.render_rgb(avatar_out, camera, background=background)

        # SH layout is [N, K, 3]; semantic is [N, C] — cannot concat for one gsplat pass.
        if packed["colors"].ndim == 3:
            rgb_out = self.render_rgb(avatar_out, camera, background=background)
            sem_out = self.render_features(avatar_out, camera)
            return {
                **rgb_out,
                "semantic": sem_out["semantic"],
                "semantic_alpha": sem_out.get("alpha", rgb_out["alpha"]),
            }

        device = packed["means"].device
        bg_rgb = self.background_rgb.to(device)
        if background is not None:
            bg_rgb = background.to(device).reshape(-1)[:3]
        k_sem = sem.shape[-1]
        feat = torch.cat([packed["colors"], sem], dim=-1)
        colors, alphas, meta = self._rasterize(packed, camera, feat)
        img, alpha = self._to_nchw(colors, alphas)
        rgb = img[:, :3]
        sem_img = img[:, 3 : 3 + k_sem]
        rgb = self._composite_background(
            rgb, alpha, self._bg_nchw(bg_rgb, 3, device, rgb.dtype)
        )
        sem_bg = torch.zeros(k_sem, device=device, dtype=sem_img.dtype)
        sem_img = self._composite_background(
            sem_img, alpha, self._bg_nchw(sem_bg, k_sem, device, sem_img.dtype)
        )
        return {
            "rgb": rgb.clamp(0, 1),
            "semantic": sem_img,
            "alpha": alpha,
            "semantic_alpha": alpha,
            "meta": meta,
            "viewspace_points": meta.get("means2d"),
            "radii": meta.get("radii"),
        }

    def render_depth(self, avatar_out, camera, render_mode="ED"):
        packed = pack_gaussians(avatar_out, sh_degree=self.sh_degree)
        device = packed["means"].device
        colors, alphas, meta = self._rasterize(
            packed, camera, packed["colors"], render_mode=render_mode
        )
        depth, alpha = self._to_nchw(colors, alphas)
        return {"depth": depth, "alpha": alpha, "meta": meta}

    def render_features(self, avatar_out, camera, features=None, backgrounds=None):
        packed = pack_gaussians(avatar_out, rgb_activation=None, sh_degree=self.sh_degree)
        features = features if features is not None else packed["sem_features"]
        K = features.shape[-1]
        device = features.device
        if backgrounds is None:
            backgrounds = torch.zeros(K, device=device, dtype=features.dtype)
        colors, alphas, meta = self._rasterize(packed, camera, features)
        sem, alpha = self._to_nchw(colors, alphas)
        # bg=0 → ``semantic`` is alpha-blended accum (Σ w_i f_i); use for seg L2, not accum/alpha.
        sem = self._composite_background(sem, alpha, self._bg_nchw(backgrounds, K, device, sem.dtype))
        return {
            "semantic": sem,
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
        packed = pack_gaussians(avatar_out, rgb_activation=None, sh_degree=self.sh_degree)
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

    def render_full_face_region(self, avatar_out, camera, ict_faces, ict):
        """
        Render ICT full-face attribution vs other surface Gaussians.

        Channel 0 = skin-face (4) + eye-occlusion (6) Gaussians only.
        """
        from utils.ict_regions import (
            FULL_FACE_ATTRIBUTION_CODES,
            classify_surface_triangles_batch,
        )

        surf = avatar_out["surface"]
        codes = classify_surface_triangles_batch(
            surf["face_idx"], ict_faces, ict, surf["face_idx"].device
        )
        code_tbl = torch.tensor(
            sorted(FULL_FACE_ATTRIBUTION_CODES), device=codes.device, dtype=codes.dtype
        )
        in_region = torch.isin(codes, code_tbl)
        signal = torch.stack([in_region.float(), (~in_region).float()], dim=-1)
        return self.render_expected_signal(avatar_out, camera, signal, signal_dim=2)

    def forward(self, avatar_out, camera, render_semantic=False, background=None, composite=True):
        if render_semantic and avatar_out.get("sem_features") is not None:
            return self.render_rgb_and_semantic(
                avatar_out, camera, background=background
            )
        return self.render_rgb(
            avatar_out, camera, background=background, composite=composite
        )
