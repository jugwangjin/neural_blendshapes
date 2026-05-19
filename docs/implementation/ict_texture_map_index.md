# ICT texture space (usemtl + UV local) — schema v6

## Texture image 선택 (`usemtl`)

`generic_neutral_mesh.obj`의 `usemtl` → quad → tri (동일 keep mask).

| `face_texture_map_id` | `material_names[]` | Runtime |
|----------------------|-------------------|---------|
| 0 | M_Face | **사용** |
| 1 | M_BackHead | **사용** |
| 2 | M_GumsTongue | **사용** |
| 3 | M_ScleraLeft | **사용** |
| … | M_ScleraRight, M_Teeth, M_IrisLeft, … | 매핑만 (미사용 가능) |

`primary_texture_materials`: `M_Face`, `M_BackHead`, `M_GumsTongue`, `M_ScleraLeft`

OBJ에만 있는 material은 catalog **끝에** 추가.

## UV local (atlas tile)

Texture **이미지**는 usemtl, **좌표**는 여전히 face-level atlas tile:

```text
infer_face_uv_tiles → triangle_uv_local → build_uv_seam_mesh
```

`u=1.0` 경계 fold 버그 방지용. `texture_map_tile` / `face_uv_tile_*`는 UV 보조.

## Geometry chart (별도)

| Key | 의미 |
|-----|------|
| `face_geometry_chart_id` | UV→3D lift (`vertex_parts` 기준) |
| `face_texture_map_id` | texture image sample (`usemtl`) |

Eye: texture는 material별, geometry chart는 L/R eyeball part 분리.

## Bake

```bash
python processing/ict_facekit_to_npy_full_head.py --export_uv_debug
```

## 샘플링

```python
tex_id = ict.face_texture_map_id[f]
mat = ict.material_names[tex_id]   # e.g. M_Face
uv = ict.uvs[ict.uv_faces[f, c]]   # local [0,1]
chart = ict.face_geometry_chart_id[f]
```
