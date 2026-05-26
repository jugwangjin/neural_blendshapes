# assets/default_camera.npz

**Canonical bake:** `processing/compute_camera_for_metrical_crop.py` (not `scripts/bake_default_camera.py`).

See [camera_metrical_crop.md](camera_metrical_crop.md) for metrics, NPZ keys, and `load_training_camera` behavior.

```bash
python processing/compute_camera_for_metrical_crop.py --apply-train-view --write-npz
```

`config.py` → `camera_npz = assets/default_camera.npz`. `train.py` calls `utils.camera.load_training_camera`.

Optional: seed `K_mean` / `R_mean` from `processing/average_dataset_cameras.py` before the metrical crop pass if npz is missing.
