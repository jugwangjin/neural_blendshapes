# Expression deformer + tracker personalization

## Tracker (`model/tracker_mlp.py`)

Gamma scales ICT expression intensity after MP→ICT gather.

```text
C_raw = clamp(mp_blendshape)
I_raw = mp[:, mediapipe_to_ict]   # [B, 53] ICT FaceKit coeffs
# Default (gamma_symmetric_log=True): reciprocal pair [1/γ_max, γ_max], center 1
t = 2*sigmoid(raw) - 1   # ∈ [-1, 1]
gamma = exp(log(γ_max) * t)
C_eff = I_raw ** gamma
```

Legacy affine: `gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(raw)` when
`gamma_symmetric_log=False`.

Defaults: `gamma_max=2.0` → γ ∈ [0.5, 2.0]; prior `mean(log(gamma)^2)` is symmetric
about 1. Why: `x^g` and `x^(1/g)` are inverses in multiplicative effect; affine
[0.4, 2.5] around 1 was not symmetric in log-space.

## Support-gated expression deformer (`model/expression_deform_mlp.py`)

Not a global `MLP(x, coeffs) -> [B,V,3]`.

Per AU `j`:

1. Precompute ICT mode magnitude `mag_j[v]` and soft support from `expression_shape_modes`
2. Dilate support 2 rings on mesh adjacency
3. `gate_j = support_j * expr_region_weight` (hair/accessory = 0)
4. `delta_j = gate_j * (ratio * mag_j + floor) * tanh(MLP(canonical_xyz, au_embed_j))`
5. `V += sum_j C_eff[j] * delta_j`

Weak losses only: `expr_neutral`, `expr_leak`, `expr_amp`.

## Mouth coeff debug (stage 2 early)

When ``mouth_debug_enabled`` and ``stage_local <= mouth_debug_stage_local_max``,
``train.py`` prints a line if **both** MP ``jawOpen`` and ``mouthClose`` are below
``mouth_debug_jaw_open_max`` / ``mouth_debug_mouth_close_max`` (default 0.15 / 0.35):

MP max, ICT ``coeffs`` mean, ``gamma`` mean, optional ``coeffs_raw`` for jaw/mouthClose slots.

## Training order

- Stage 1: expression deformer **off**; tracker + template deformer align coarse geometry
- Stage 2: expression deformer **on**; template/tracker frozen

## Data

`VideoDataset(au_active_boost=True)`: 30% samples from high-AU frames (`max(mp_blendshape) > 0.12`).
