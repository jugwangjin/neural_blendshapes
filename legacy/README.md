# Archived code (not used by `train.py`)

| Directory | Contents |
|-----------|----------|
| `flare/` | FLARE neural-shader stack: `test.py`, `load_nbshapes.py`, `arguments.py`, old `processing/*`, `configs/` |
| `eye_uv_slide/` | UV-slide eye Gaussians, `gaze_uv.py`, barriers |

**Active stack** (repo root): `train.py`, `model/`, `training/`, `rendering/`, `losses/train_losses.py`, `dataset/image_dataset.py`, `processing/ict_mediapipe_lmk/`, `processing/ict_facekit_to_npy_full_head.py`.

Run archive (once, on server):

```bash
python scripts/archive_flare_legacy.py
```
