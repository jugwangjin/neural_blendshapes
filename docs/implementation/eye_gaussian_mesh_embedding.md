# Eye Gaussians: mesh embedding (not UV lookup)

## Bug (2025-05)

Init sampled many `(face_idx, bary)` on the forward eyeball hemisphere, converted to atlas `uv`, then **at render** called `surface_points_from_uvh` → `uv_to_face_bary(uv, UVMesh)`.

ICT `uvs` are **seam/atlas** coordinates. Almost all init `uv` points **miss** the active triangle set; fallback picks the **nearest triangle centroid** → almost always the sclera chart center (pupil) → one white dot per eye.

## Fix

Same as surface Gaussians:

1. `sample_sclera_layout()` → `(uv, face_idx, bary)` on L/R eyeball forward hemisphere (area-weighted).
2. `EyeTextureGaussians` registers frozen `face_idx_left/right`, `bary_left/right`.
3. Forward: `xyz = sample_surface(posed_verts, faces, face_idx, bary)`.
4. `uv` buffer kept for gaze / barrier losses only (does not drive 3D position until gaze reprojection is implemented).

**Recreate avatar** after this change (`GaussianAvatar.from_ict`).
