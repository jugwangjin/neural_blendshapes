# Camera view correction (FLAME-aligned ICT)

## Convention

- **World (ICT after `flame_alignment`)**: FLAME-style, face roughly toward **-Z**; `from_mesh_bounds` places the camera on **+Z** looking at the centroid → back of head, often upside-down in the image.
- **Camera (OpenCV / gsplat)**: +X right, +Y down, **+Z forward** (into the scene).
- **Extrinsics (row vectors)**: `p_cam = p_world @ R.T + t`.

## Default correction

`FixedCamera.with_view_correction(pivot, yaw_deg=180, roll_deg=180)`:

1. **Yaw 180°** about world **+Y** at mesh pivot (`with_azimuth_y`) — camera orbits to the face side.
2. **Roll 180°** about camera **+Z** (`with_roll_forward_deg`) — upright image.

Applied in `debug/sanity_gaussian_layout.py` (via baked npz) and `train.py` via `load_training_camera`.

`assets/default_camera.npz` from `bake_default_camera.py` is **uncorrected** mesh-bounds fit; do not bake the 180°/180° into the npz if runtime correction is enabled (avoids double application).

## Head yaw (constant camera)

Orbit is **not** applied to the camera. Sanity sweeps ``head_yaw_deg`` via ``ICTDeformer.apply_head_yaw`` with ``pose_weight_fixed=1.0``:

```bash
python debug/sanity_gaussian_layout.py --sweep-yaw -30,0,30
```

## Compare old renders

```bash
python debug/sanity_gaussian_layout.py
```
