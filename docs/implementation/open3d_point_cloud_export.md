# Open3D colored point cloud export

## Summary

Open3D can read/write colored point clouds via `o3d.geometry.PointCloud` with per-point RGB in `[0, 1]`. This repo wraps that in `utils/export_open3d.py` and hooks it into `scripts/sanity_gaussian_layout.py`.

## API (`utils/export_open3d.py`)

| Function | Input | Output |
|----------|--------|--------|
| `save_colored_point_cloud(path, xyz, rgb, ...)` | `[N,3]` positions; `[N,3]` RGB in `[0,1]` or logits | `.ply` (binary) |
| `save_gaussian_point_cloud(path, avatar_out, ...)` | `avatar_out["xyz"]`, `avatar_out["color"]` from `GaussianAvatar` | same |
| `save_mesh_point_cloud(path, verts, rgb, ...)` | deformed mesh `[B,V,3]` or `[V,3]` + per-vertex RGB | same |
| `mesh_vertex_colors_from_regions(ict, region_rgb, device)` | ICT region indices + palette dict | `[V,3]` |

- Logit colors (Gaussian `color` params) are auto-mapped with `sigmoid` when values fall outside `[0, 1]`.
- `max_points` + `seed` random-subsample large clouds (default 100k).

## Sanity script

Default: PLY export **on** for each rendered frame.

```bash
python scripts/sanity_gaussian_layout.py --out debugs/sanity_gaussians
# → jaw_*_yaw*_gaussians.ply (Gaussian centers + region/opacity colors)
# → optional mesh: --pcd-mode mesh|both
python scripts/sanity_gaussian_layout.py --no-save-pcd
```

View in Open3D GUI or MeshLab:

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("debugs/sanity_gaussians/jaw_0.000_yaw+0_gaussians.ply")
o3d.visualization.draw_geometries([pcd])
```

## Dependencies

`open3d` (already used in `processing/`, `gui_by_facs.py`, etc.). Lazy-import inside `save_colored_point_cloud` so training code paths without Open3D are unaffected until export is called.
