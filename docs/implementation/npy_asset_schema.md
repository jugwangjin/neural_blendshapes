# ICT npy asset schema (runtime source of truth)

Official [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) README ranges are **reference only**.
Training/runtime use `assets/ict_facekit_torch.npy` built by:

```bash
python processing/ict_facekit_to_npy_full_head.py
```

## Variant: `full_head_24591` (schema v2)

- Vertices **0:24591** — official quad parts **#0–#8** only.
- **Excluded** from asset: parts #9–#16 (lacrimal, eye blend, occlusion, eyelashes).
- **Eyeballs only**: `21451:23021` (L), `23021:24591` (R).

### Part ids (`vertex_parts`)

| id | region | vertex range |
|----|--------|----------------|
| 0 | face skin | 0:9409 |
| 1 | head/neck | 9409:11248 |
| 2 | mouth socket | 11248:13294 |
| 3 | eye socket L | 13294:13678 |
| 4 | eye socket R | 13678:14062 |
| 5 | gums/tongue | 14062:17039 |
| 6 | teeth | 17039:21451 |
| 7 | eyeball L | 21451:23021 |
| 8 | eyeball R | 23021:24591 |

### Index arrays (source of truth)

| key | use |
|-----|-----|
| `face_indices` | legacy face+mouth+teeth union |
| `not_face_indices` | head/neck |
| `eyeball_indices` | L+R eyeball only |
| `surface_sample_vertex_indices` | surface Gaussians (no teeth, no eyeball) |
| `mouth_interior_vertex_indices` | gums/tongue — higher sampling density |
| `teeth_indices` | excluded from surface sampling |
| `left_eyeball_indices` / `right_eyeball_indices` | eye UV texture space |

**No hair/accessory** in ICT topology — accessory = optional free Gaussians from segmentation.

## Gaussian sampling (`utils/sampling.py`)

- Surface: triangles with all verts in `surface_sample_vertex_indices`.
- Skip triangles touching `teeth_indices` or `eyeball_indices`.
- Per-triangle `k`: face/head/mouth_socket = 4 (config), mouth_interior = 8.
- Eyes: `EyeTextureGaussians` on eyeball triangles only (`utils/texture_spaces.py`).

## Semantics (`rendering/gaussian_semantics.py`)

`vertex_semantic_name(ict, v)` uses index membership, not `part_id → hair/accessory`.

## Regenerate

After changing processing script, rerun on server (WSL/docker) and commit or deploy the new `.npy`.
