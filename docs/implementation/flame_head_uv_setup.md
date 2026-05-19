# `assets/flame_head_uv.obj` 준비

## 이 파일이 뭔가

| 파일 | 내용 |
|------|------|
| `assets/canonical_eye_smpl.obj` | FLARE **processed** topology, **vt 없음** (면만) |
| `assets/flame_head_uv.obj` | **같은 면 개수** + `vt` + `f v/vt` (UV texture QA용) |
| `assets/flame_uv_for_pkl_faces.npz` | flare UV → pkl `generic_model.pkl` 면 순서로 remap 캐시 (자동 생성) |

직접 Blender로 “새로 만드는” 게 아니라, **이미 공개된 FLAME/FLARE UV 헤드 메쉬를 복사**하는 작업입니다.

## 방법 1 — metrical-tracker 설치 (가장 쉬움)

이미 `processing/metrical-tracker` 쓰는 경우:

```bash
cd processing/metrical-tracker
bash install.sh   # FLAME 계정 + mesh.zip 등 다운로드
cd ../..

python processing/flame/prepare_flame_head_uv.py --build_cache
```

스크립트가 아래 등을 찾아 `assets/flame_head_uv.obj`로 복사합니다.

- `metrical-tracker/data/mesh/head_template_mesh.obj`
- `metrical-tracker/flame/geometry/head_template_mesh.obj`

## 방법 2 — FLAME 공식 다운로드

[flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) 에서:

1. `FLAME2020.zip` → `processing/flame/FLAME2020/generic_model.pkl` (이미 있음)
2. `TextureSpace.zip` (또는 metrical `install.sh`가 받는 texture/mesh 패키지)

압축 안의 `head_template_mesh.obj` / `head_template.obj` 처럼 **`vt`가 있는 OBJ**를 찾아:

```bash
python processing/flame/prepare_flame_head_uv.py \
  --src /path/to/head_template_mesh.obj \
  --build_cache
```

## 방법 3 — 이미 있는 경로만 지정

```bash
python processing/flame/prepare_flame_head_uv.py --src /path/to/head_with_vt.obj --build_cache
```

## 검증

```bash
# flare(8090대) vs pkl(9976) 면 대응 통계
python processing/flame/FLAME2020/uvs_for_flame.py

# 후보 경로만 나열
python processing/flame/prepare_flame_head_uv.py --list_only
```

성공 후 bake:

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
# → debug/ict_mediapipe_texture.png, flame_mediapipe_texture.png, mediapipe_landmarks_ict_vs_flame.png
```

## topology 맞는지

`prepare_flame_head_uv.py`는 다음 중 하나면 통과:

- UV face 수 == `canonical_eye_smpl.obj` face 수 → 그대로 사용
- UV face 수 == pkl face 수 → 그대로 사용  
- UV face 수 == flare 수이고 pkl과 face identity remap 가능 → `flame_uv_for_pkl_faces.npz` 생성

DECA `head_template.obj` 등 **다른 topology**면 face 수가 안 맞아 실패합니다. 그때는 metrical/FLAME TextureSpace 쪽 mesh를 쓰세요.
