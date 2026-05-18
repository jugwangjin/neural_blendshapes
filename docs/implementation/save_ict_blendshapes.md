# save_ict_blendshapes.py

ICT-FaceKit expression blendshape 데이터를 `flare/utils/ict_model.py`의 `ICTFaceKitTorch`에서 추출해 저장한다.

## 실행

프로젝트 루트에서:

```bash
python save_ict_blendshapes.py
python save_ict_blendshapes.py --save_meshes
python save_ict_blendshapes.py --save_meshes --to_canonical
```

## 출력 (`ict_bshapes/`)

| 파일 | 내용 |
|------|------|
| `ict_blendshapes.npz` | `neutral_mesh` (V,3), `expression_shape_modes` (E,V,3), `faces`, `expression_names`, `num_expression` |
| `expression_names.txt` | 인덱스–이름 목록 |
| `meshes/neutral.obj` | 중립 메쉬 (`--save_meshes`) |
| `meshes/{name}.obj` | 해당 expression만 `weight`로 활성화한 메쉬 (`forward`, 눈동자 회전 포함) |

## 메모

- NPZ의 `expression_shape_modes`는 `ICTFaceKitTorch` 버퍼와 동일한 선형 델타다.
- OBJ는 `forward()` 결과이므로 eye look 계열은 회전 보정이 반영된다.
- 기본은 model space (`--to_canonical` 미사용). canonical 공간이 필요하면 플래그 사용.
