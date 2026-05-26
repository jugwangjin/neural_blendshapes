# Neural Blendshapes (unit-wise ICT + 3DGS)

MediaPipe → tracker MLP → ICT deformation → **surface Gaussians** → **gsplat**.

## Active repo (use these)

| Area | Path |
|------|------|
| Train | `train.py`, `config.py` |
| Stages / optim | `training/` |
| Model | `model/ict_model.py`, `model/ict_deformer.py`, `model/tracker_mlp.py`, `model/gaussian_avatar.py` |
| Render | `rendering/` |
| Losses | `losses/train_losses.py`, `losses/mediapipe_landmark_478.py` |
| Data | `dataset/image_dataset.py` (`dataset_type=flare` = **folder layout**, not FLARE shader code) |
| Bake | `processing/ict_facekit_to_npy_full_head.py`, `processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py` |
| Camera | `processing/compute_camera_for_metrical_crop.py` → `assets/default_camera.npz` |
| Debug | `debug/sanity_train_stack.py`, `debug/verify_mp_onehot_ict_render.py` |

## Setup

```bash
# 1) ICT npy + MP→ICT landmark embedding
python processing/ict_facekit_to_npy_full_head.py
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py --iris-bake ray_occ

# 2) Training camera (metrical-tracker crop framing → assets/default_camera.npz)
python processing/compute_camera_for_metrical_crop.py --apply-train-view --write-npz

# 3) Sanity (optional)
python debug/sanity_train_stack.py --check all

# 4) Train (edit config.py input_dir first)
python train.py
```

Assets: `assets/ict_facekit_torch.npy`, `assets/mediapipe_name_to_indices.pkl`,  
`assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz`,  
`assets/default_camera.npz` (from `compute_camera_for_metrical_crop.py`, not `scripts/bake_default_camera.py`)

## Legacy (do not mix with train)

Old FLARE / neural-shader / UV-slide code lives under **`legacy/`**:

- `legacy/flare/` — `test.py`, `load_nbshapes.py`, `arguments.py`, old configs, …
- `legacy/eye_uv_slide/` — UV-slide eye Gaussians

**One-time cleanup on server** (moves remaining root FLARE scripts + `configs/`):

```bash
python scripts/archive_flare_legacy.py
```

Index: `legacy/README.md`, `docs/implementation/project_layout.md`
