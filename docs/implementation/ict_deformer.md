# ICTDeformer (unified template + expression)

## Field model (canonical xyz)

Template and expression correctives are **the same style of field**: MLP inputs are
`neutral_mesh_canonical` positions. Outputs apply to **all vertices except teeth** (hard off).

| Region | Forward | Regularization (`deform_reg_weight`) |
|--------|---------|-------------------------------------|
| Face / eyeball | Full field | Low (0.02–0.05) — lids and sclera move together |
| Eye socket (orbit) | Full field | High (1.0) — penalize large deltas behind the eye |
| Teeth | Zero | High |

Hard `gate=0` on eyeballs was removed: it caused seams when lids deformed but sclera did not.

Expression still uses support-gated AU magnitudes; `build_expr_region_weight` now allows
eyeball/socket (1.0). Leak/amp/neutral losses unchanged; `expr_socket` adds weighted L2 on
`expression_delta` in eye-socket regions.

## Eye Gaussians (front hemisphere)

`sample_sclera_uv` uses `sclera_front_face_indices`: `M_Sclera* ∩ eyeball` triangles whose
face normal satisfies `n · forward ≥ -0.15`, where `forward` is eye-center → sclera chart
pole in 3D. This is slightly wider than a strict front hemisphere for training views.

Previously: all `M_Sclera*` faces (could include wrap-around on the eyeball).

## Training

```python
deformer = ICTDeformer(ict, build_expr_region_weight(ict), n_coeffs=52)
```

- `w_template_smooth` → `template_regularization_loss()` (weighted L2)
- `w_expr_amp` also weights `expr_socket` when expression deformer is active
