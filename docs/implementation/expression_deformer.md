# Expression deformer + tracker personalization

## Tracker (`model/tracker_mlp.py`)

Gamma scales ICT expression intensity after MP→ICT gather.

```text
C_raw = clamp(mp_blendshape)
I_raw = mp[:, mediapipe_to_ict]   # [B, 53] ICT FaceKit coeffs
gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(raw_gamma)
C_eff = I_raw ** gamma
```

Defaults: `gamma_min=0.4`, `gamma_max=2.5`. Prior: `mean(log(gamma)^2)`.

## Support-gated expression deformer (`model/expression_deform_mlp.py`)

Not a global `MLP(x, coeffs) -> [B,V,3]`.

Per AU `j`:

1. Precompute ICT mode magnitude `mag_j[v]` and soft support from `expression_shape_modes`
2. Dilate support 2 rings on mesh adjacency
3. `gate_j = support_j * expr_region_weight` (hair/accessory = 0)
4. `delta_j = gate_j * (ratio * mag_j + floor) * tanh(MLP(canonical_xyz, au_embed_j))`
5. `V += sum_j C_eff[j] * delta_j`

Weak losses only: `expr_neutral`, `expr_leak`, `expr_amp`.

## Training order

- Stage 1: expression deformer **off**; tracker + template deformer align coarse geometry
- Stage 2: expression deformer **on**; template/tracker frozen

## Data

`VideoDataset(au_active_boost=True)`: 30% samples from high-AU frames (`max(mp_blendshape) > 0.12`).
