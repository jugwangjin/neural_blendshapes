# Debug / verification code

Model-unrelated scripts and helpers live here — not imported by `train.py` or the training stack.

## Scripts

| Script | Purpose |
|--------|---------|
| `sanity_gaussian_layout.py` | Region-colored Gaussian layout renders (jaw/yaw/gaze sweeps) |
| `sanity_train_stack.py` | Quick compile / forward / render / loss smoke tests |
| `verify_mp_onehot_ict_render.py` | MP 52 one-hot → ICT scatter → front PNG per expression name |

## Library (`sanity/`)

| Module | Purpose |
|--------|---------|
| `region_colors.py` | Debug region palette for sanity renders |
| `depth_vis.py` | Depth colormap + overlay for debug PNGs |
| `export_open3d.py` | Colored PLY export |

## Outputs

Default render output: `debug/out/sanity_gaussians/`, MP one-hot QA: `debug/out/mp_onehot_ict/`

## Run (repo root)

```bash
python debug/sanity_gaussian_layout.py
python debug/sanity_train_stack.py --check all
python debug/verify_mp_onehot_ict_render.py
```

See `debug/docs/` for detailed notes.
