# ICT ↔ FLAME alignment (coarse + rigid)

## Two-stage fit

### 1) `ict_facekit_to_npy_full_head.py` — coarse (no R)

Grid-search `jawOpen` on neutral ICT, then fit `s,T` only:

- ICT: `neutral + jawOpen * w`
- FLAME: canonical pose, `flame_static_embedding.pkl` 68 pts
- Pair: `[landmark_start:]` index order (default 17)

Stored in `ict_facekit_torch.npy`:

| Key | Role |
|-----|------|
| `flame_similarity_ict_jaw_open` | Optimized jaw weight |
| `flame_similarity_s`, `flame_similarity_T` | Coarse scale + translation |
| `flame_similarity_lmk_err_*` | Diagnostics |

```bash
python processing/ict_facekit_to_npy_full_head.py
# --no_optimize_jaw  --ict_jaw_open 0.75  --jaw_min 0 --jaw_max 1.2
```

### 2) `bake_mediapipe_to_ict.py` — rigid `s,R,T`

On landmarks after coarse+jaw mesh:

- `pytorch3d.ops.corresponding_points_alignment` → `s1,R1,T1`
- Compose with coarse: `s_tot, R_tot, T_tot = compose(s0,t0, s1,R1,t1)`
- Write `flame_alignment_s`, `flame_alignment_R`, `flame_alignment_T` back to npy (default)

```bash
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
# --skip_save_alignment  # bake only, do not overwrite npy
```

## Runtime (`model/ict_model.py`)

`forward(..., apply_flame_similarity=True)`:

1. `neutral + expr + idt` (local)
2. If `flame_alignment_*` in npy: `s * (mesh @ R) + T`
3. Else fallback: `flame_similarity_s * mesh + T`
4. Optional `to_canonical_space` via `ict_identity.npy` `s,R,T` (separate)

Canonical expression jaw uses `flame_similarity_ict_jaw_open` from npy.

## Transform convention

Row vectors: `x_flame = s * (x_ict @ R) + T`

Matches `optimize_ict_expression_to_flame.py` / pytorch3d alignment.
