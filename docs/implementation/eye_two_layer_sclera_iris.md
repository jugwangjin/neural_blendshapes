# ICT eyeball: two-layer mesh, NICP, and UV QA

## Geometry / materials

ICT eyeball is **not** a single surface in the texture pipeline:

| Layer | Material | UV chart | Role |
|-------|----------|----------|------|
| Outer disk | `M_ScleraLeft` / `M_ScleraRight` | Filled disk, local UV | Eye Gaussians, iris MP landmarks (468–477) |
| Inner annulus | `M_IrisLeft` / `M_IrisRight` | Separate island (corner of atlas) | Iris albedo ring; **empty center** at pupil |

`utils/eye_chart.py` documents this split. `sclera_chart_center_bary()` maps chart center **(0.5, 0.5)** to the **pupil pole** on the sclera disk.

## Why full-disk UV PNG looks “one point”

If 3D iris landmarks sit near the pupil on the sclera shell, their `triangle_uv_local` values cluster near **(0.5, 0.5)**. On a 2048×2048 full-disk wireframe that is **a few pixels** — not necessarily wrong 3D.

Check:

1. Bake log: `iris UV on M_Sclera*` — `dist(0.5,0.5)` and `u/v` ranges.
2. **`ict_eyeball_{left,right}_iris5_texture_zoom.png`** — magnified pentagon QA.
3. 3D spread lines: `fitted iris spread` in `eye_transplant.py`.

## Two-layer hypothesis (NICP error)

Mapping **FLAME single-layer eyeball** → **full ICT eyeball** (sclera + iris annulus + shared verts) can bias chamfer:

- `M_Iris*` tris add an **inner shell** near the pupil.
- Chamfer pulls FLAME verts toward that shell, then projection onto `M_Sclera*` can still look OK in transfer distance but mis-place iris in UV/3D.

### Eye alignment (current)

**No vertex NICP on eyeball.** Per-eye **`s,T` only** (`R=I`; `eye_rigid_align.py`):

- **Chamfer**: `pytorch3d.loss.chamfer_distance` (`single_directional=False`) on full FLAME eyeball (≈546 V) ↔ ICT sclera verts (≈770 V) — **no subsampling**; unequal counts are valid.
- **Init `s,T`**: front/back anchor pairs only (`fit_uniform_scale_translation`, N=2).
- **Anchors (2 pairs, front–back)**:
  - **Front**: FLAME MP iris center (468/473) ↔ ICT sclera UV **(0.5, 0.5)** pole.
  - **Back**: same axis through eyeball centroid — `back = 2·center − front`, snapped to nearest eyeball vertex (FLAME submesh / ICT `left|right_eyeball_indices`).
- Fitted FLAME iris pentagon → project onto `M_Sclera*` for MP 468–477 embedding.

CLI: `--eye_rigid_iters`, `--eye_rigid_lr`, `--eye_w_chamfer`, `--eye_w_anchor`.

4. **UV export**: `*_iris5_texture_zoom.png` in `texture_viz.export_eyeball_iris5_texture`.

## Small circle in corner of PNG

Separate UV island — typically **`M_Iris*`** annulus chart, not used for iris MP bake. Iris landmarks are intentionally on the **large sclera disk**, not the annulus island.

## Re-run

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```

Inspect `processing/ict_mediapipe_lmk/debug/texture_maps/ict_eyeball_*_iris5_texture_zoom.png`.

## Eye fitting debug meshes

After bake (unless `--no_export_debug`), per-eye exports under `debug/eyes/`:

| File | Content |
|------|---------|
| `flame_eye_{left,right}_fitted.obj` | FLAME eyeball after eye rigid `s,R,T` |
| `flame_eye_{left,right}_canonical.obj` | Before per-eye rigid |
| `*_mp_iris_center.ply` / `*_sclera_uv_center.ply` | Rigid anchor pair |
| `ict_eyeball_{left,right}.obj` | ICT-FaceKit eyeball submesh (NICP target) |
| `ict_eye_fitting.npz` | Vertices/faces/global indices |

Load fitted FLAME + ICT eyeball in the same viewer (both in FLAME space) to compare alignment.
