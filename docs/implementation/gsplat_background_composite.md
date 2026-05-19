# gsplat backgrounds workaround

## Problem

With `gsplat_packed=True` and `backgrounds` passed into `rasterization()`, gsplat 1.5.x hits:

```
AssertionError: backgrounds.shape == image_dims + (channels,)
# got torch.Size([1, 3])
```

See [gsplat#764](https://github.com/nerfstudio-project/gsplat/issues/764).

## Fix (`rendering/avatar_renderer.py`)

- Always call `rasterization(..., backgrounds=None, packed=cfg.gsplat_packed)`.
- Composite in Python after rasterize:

  `rgb = fg + (1 - alpha) * bg`  with `bg` shaped `[1, C, 1, 1]`.

Black background (sanity default) is unchanged. Non-black `background=` / `bg_color` still work via post-composite.
