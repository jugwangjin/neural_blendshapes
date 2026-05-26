# Gaussian scale KNN initialization

## Method (3DGS / gsplat style)

At avatar construction, per-Gaussian:

1. World position `xyz` (surface bary sample on ICT mesh).
2. Mean distance `d` to `k` nearest neighbors (default `k=3`, excluding self).
3. `log_scale = log(d × factor)` replicated to 3 axes; training uses `scale = exp(log_scale)`.

Backend order:

1. `simple_knn._C.distCUDA2` if installed (original Inria 3DGS CUDA).
2. Else `scipy.spatial.cKDTree` on CPU (one-time at init).

## Config

```python
gaussian_scale_knn_k: int = 3
gaussian_scale_knn_factor: float = 1.0
```

## Code

- `utils/gaussian_scale_init.py` — `knn_mean_distance`, `log_scale_from_knn`, `init_module_log_scale`
- `GaussianAvatar.from_ict` → `_init_knn_scales`

Re-build avatar / restart training after changing `k` or `factor`.
