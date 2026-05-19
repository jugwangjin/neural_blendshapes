# ICT FaceKit → `assets/ict_facekit_torch.npy`

런타임 topology·region index의 **source of truth**.

## 선행: ICT_FaceKit (gitignored)

프로젝트 루트에 **공식 ICT-FaceKit**을 `ICT_FaceKit/` 이름으로 clone (또는 `ICT_FACEKIT_ROOT` 환경변수):

```bash
cd /path/to/neural_blendshapes_unit_wise
git clone https://github.com/USC-ICT/ICT-FaceKit.git ICT_FaceKit
# FaceXModel/ 가 포함되어 있어야 함
```

필수 경로: `ICT_FaceKit/Scripts/face_model_io.py`, `ICT_FaceKit/Scripts/ict_face_model.py`, `ICT_FaceKit/FaceXModel/generic_neutral_mesh.obj`

`face_model_io`는 `import ict_face_model`을 쓰므로 `processing/paths.py`가 `ICT_FaceKit/Scripts`도 `sys.path`에 넣는다.

## 생성

```bash
# 권장 (공식 parts #0–#8, V=24591)
python processing/ict_facekit_to_npy_full_head.py

# 동일 스키마 (별칭 스크립트)
python processing/ict_facekit_to_npy.py
```

출력: `assets/ict_facekit_torch.npy`

## `vertex_parts` (공식 ICT)

| id | range | region |
|----|-------|--------|
| 0 | 0:9409 | face skin |
| 1 | 9409:11248 | head/neck |
| 2 | 11248:13294 | mouth socket |
| 3 | 13294:13678 | eye socket L |
| 4 | 13678:14062 | eye socket R |
| 5 | 14062:17039 | gums/tongue |
| 6 | 17039:21451 | teeth |
| 7 | 21451:23021 | eyeball L |
| 8 | 23021:24591 | eyeball R |

Parts #9–#16 (lacrimal, lashes, …)는 **제외**.

## npy 주요 keys

| Key | shape | 용도 |
|-----|-------|------|
| `neutral_mesh` | [V, 3] | 3D vertex positions |
| `faces` | [F, 3] | 3D triangle → vertex index |
| `uvs` | [VT, 2] | **seam-safe** local UV in [0,1) per atlas tile |
| `uv_faces` | [F, 3] | triangle → **UV vertex index** (into `uvs`) |
| `uv_tile_index_vt` | [VT, 2] int | `(tu, tv)` = `floor(uv)` before split — **texture map tile** in atlas grid |
| `uv_neutral_mesh` | [V, 2] | 3D vertex당 local UV (seam은 첫 corner만) |
| `uv_tile_index_v` | [V, 2] int | per-3D-vertex atlas tile index |
| `vmapping` | [VT] | `vmapping[i]` = UV vert i의 원본 3D vertex id |
| `vertex_parts` | [V] | mesh part id 0–8 (topology); tile index와 상관 있으나 동일하지 않음 |
| `left_eyeball_indices`, … | lists | region / eye texture |
| `asset_variant`, `asset_schema_version` | scalar | 호환성 |

런타임 UV lookup (`uv_to_face_bary`)은 **`uvs` + `uv_faces`** 를 쓴다.  
`uv_neutral_mesh`는 per-3D-vertex UV; seam에서 여러 chart가 있어도 하나만 저장된다.

## Texture / geometry charts (schema v6)

상세: [ict_texture_map_index.md](ict_texture_map_index.md)

```text
usemtl → face_texture_map_id / material_names
face atlas tile → triangle_uv_local → seam uvs
vertex_parts → face_geometry_chart_id
```

Runtime textures: `M_Face`, `M_BackHead`, `M_GumsTongue`, `M_ScleraLeft` (+ 전 material catalog).

## UV 시각화 (bake 시)

```bash
python processing/ict_facekit_to_npy_full_head.py --export_uv_debug
# → debug/ict_facekit_uv/
#    ict_part_atlas.png      vertex_parts 색칠
#    ict_uv_wireframe.png    UV island 외곽
#    ict_per_vertex_uv.png   uv_neutral_mesh 점박이
#    ict_uv_seam.obj         seam mesh (Blender/MeshLab)
#    uv_indices.npz          uvs, uv_faces, vmapping, …
#    uv_index_notes.txt
```

MediaPipe landmark UV bake: `processing/ict_mediapipe_lmk/` 의 `texture_viz.py` (별도 파이프라인).

Region 빌드 로직: `processing/ict_region_dict.py` → `model/ict_model.py`, `utils/ict_regions.py`.

## 재생성 후

MediaPipe embedding도 다시 bake:

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```
