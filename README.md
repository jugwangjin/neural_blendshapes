# Neural Blendshapes (unit-wise ICT + 3DGS)

MediaPipe blendshapes → tracker MLP → ICT mesh deformation → **surface Gaussians** + **eye texture Gaussians** → **gsplat** rendering.

This repo is a refactor away from FLARE / neural mesh shader / trainable face UV. Legacy FLARE code lives under `legacy/`.

## Stack

| Component | Path |
|-----------|------|
| ICT mesh | `model/ict_model.py`, `model/ict_deformer.py` |
| Tracker | `model/tracker_mlp.py` |
| Expression deformer | `model/expression_deform_mlp.py` |
| Surface Gaussians (fixed bary) | `model/surface_gaussians.py` |
| Eye UV slide | `model/eye_texture_gaussians.py` |
| Renderer | `rendering/avatar_renderer.py` (requires `gsplat`) |
| Training | `train.py`, `training/stages.py` |

Module map: `model/README.md`. Unused FLARE helpers: `legacy/model/`.

## Setup

```bash
git submodule update --init gsplat
pip install gsplat   # or: pip install -e gsplat
pip install torch torchvision  # + project deps
```

Assets:

```bash
python processing/ict_facekit_to_npy_full_head.py
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```

→ `assets/ict_facekit_torch.npy`, `assets/ict_mediapipe_landmark_embedding_from_metrical_tracker.npz`  
Docs: `docs/implementation/ict_facekit_npy.md`, `docs/implementation/ict_mediapipe_lmk_baker.md`.

## Train

```bash
python train.py
```

Config: `config.py` (surface `k` per region, `n_eye_gaussians_per_side`, `batch_size=1`).

## ICT regions (authoritative)

From `assets/ict_facekit_torch.npy` index arrays (built by `ict_facekit_to_npy_full_head.py`). Part ids:

| id | region |
|----|--------|
| 0 | face |
| 1 | head/neck |
| 2 | mouth socket |
| 3–4 | eye sockets |
| 5 | gums/tongue |
| 6 | teeth |
| 7–8 | eyeballs L/R |

Hair/accessory are **not** in ICT; use segmentation + optional `AccessoryGaussians`.

## Legacy

- `legacy/model/` — FLARE `NeuralBlendshapes`, ResNet encoder, eye plane Gaussians
- `legacy/losses/mediapipe_landmark.py` — 68-pt gbuffer landmark loss
- Root scripts (`test.py`, `gui_by_facs.py`, …) still import `flare` — not used by `train.py`

Original FLARE paper/README content: see [FLARE project page](https://flare.is.tue.mpg.de/).
