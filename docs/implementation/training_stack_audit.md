# Training stack (audit)

See **`eye_uv_slide.md`** for eyes/gaze.

**Active:** `train.py` → `TrackerCorrectionMLP` → `ICTDeformer` → `GaussianAvatar` (surface) → `GaussianRenderer`.

No `EyeTextureGaussians`, no `utils/eye_uv_sampling` in tree root (archived under `legacy/eye_uv_slide/`).
