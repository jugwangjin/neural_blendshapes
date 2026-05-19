# Bilateral eye Gaussians — shared chart UV, per-side mesh embed

## Problem (left OK, right broken)

- **Shared:** one `uv[i]`, `color[i]`, `log_scale[i]`, … per Gaussian index `i`.
- **Not shared:** global `face_idx[i]` on the full ICT mesh. Triangle `face_idx=42` on the left eyeball is a different 3D triangle than `42` on the right.
- **Wrong fixes that caused regressions:**
  1. Reusing left `face_idx` / `bary` on right posed verts → Gaussians stick to left-eye geometry.
  2. Independent right hemisphere resample → spread on R but **no index correspondence** with shared `uv[i]`.
  3. Per-frame atlas `uvs` + `uv_to_face_bary` → many misses → centroid fallback → pupil cluster.

Texture atlases share the same **chart** parameterization (`M_ScleraLeft` / `M_ScleraRight` disks). UV order in the chart is consistent; only the **chart UV → global (face, bary)** map is side-specific.

## Correct model

| Shared per index `i` | Per side at forward |
|----------------------|---------------------|
| `uv[i]` (canonical chart, from L hemisphere sample) | `uv_eff = uv + gaze` (+ mirror U on R if `mirror_right_u`) |
| appearance params | `embed_chart_uv_on_mesh(ict, side, uv_eff)` → `(face_idx, bary)` |
| | `sample_surface(posed_verts, faces, face_idx, bary)` |

Init (`sample_shared_sclera_layout`):

1. Area-weighted hemisphere sample on **L** → `uv`, `face_idx_left`, `bary_left`.
2. **R embed only:** `embed_chart_uv_on_mesh(ict, "R", uv, mirror_right_u=…)` — do **not** resample R.

Forward (`EyeTextureGaussians._instantiate_mesh`): same chart→mesh lookup on **posed** verts every call (gaze moves 3D).

Lookup pipeline (`utils/uv_mesh.py`):

1. **UV coords → chart triangle + bary** — `uv_points_to_chart_triangle_bary(tri_local_uv)`
2. **Chart tri → mesh face** — `chart_triangle_to_mesh_face(tri_idx, mesh_face_per_tri)`
3. **Same bary on mesh** — `sample_surface(posed_verts, faces, face_idx, bary)`

Composed as `chart_uv_to_mesh_face_bary`. Best chart triangle = max min-barycentric score.

## Sanity render

`scripts/sanity_gaussian_layout.py` default `--sweep-gaze 0,0.03,-0.03` → neutral + ±U/±V offsets on **both** eyes. Disable with `--no-sweep-gaze`.

## After code changes

Recreate avatar (`GaussianAvatar.from_ict`) or reload checkpoint so `EyeTextureGaussians` layout is rebuilt.
