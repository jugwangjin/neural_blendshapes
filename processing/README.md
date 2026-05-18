# processing/

Data-prep and alignment scripts. Run from **repo root**.

## Layout

| Path | Role |
|------|------|
| `flame/` | FLAME decoder (`processing/flame/FLAME2020/generic_model.pkl`) |
| `metrical-tracker/` | MediaPipe → FLAME embedding |
| `large-steps-pytorch/` | Large Steps (NICP) |
| `ict_mediapipe_lmk/` | Bake MP landmarks onto ICT |
| `paths.py` | Shared `REPO_ROOT`, asset paths, `setup_import_paths()` |

## Imports

Scripts bootstrap `sys.path` with repo root + `processing/`, then:

- `from flame.FLAME import FLAME`
- `from model.ict_model import ICTFaceKitTorch`
- `from processing.paths import FLAME_MODEL, ASSETS_DIR, ...`

Legacy scripts still use `flare.dataset` / `flare.core` — keep a `flare/` checkout at repo root if you run them.

## Examples

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
python processing/optimize_ict_expression_to_flame.py --config ...
```
