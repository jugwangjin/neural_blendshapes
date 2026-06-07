# FLAME alignment (`s`, `R`, `T`) vs legacy canonical pose

## Npy / bake pipeline (source of truth)

Built by `processing/ict_facekit_to_npy_full_head.py` and refined in `processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py`:

| Field | Meaning |
|-------|---------|
| `flame_similarity_ict_jaw_open` | Grid-searched `jawOpen` weight on ICT neutral |
| `flame_similarity_s`, `flame_similarity_T` | Coarse uniform scale + translation (legacy `--coarse_st_only`; often identity when full rigid is stored) |
| `flame_alignment_s`, `flame_alignment_R`, `flame_alignment_T` | Single rigid ICT→FLAME map: `x' = s * (x @ R) + T` on landmarks from **jaw-open** mesh |

Convention matches `processing/ict_flame_similarity.py` and `ICTFaceKitTorch.apply_flame_similarity`.

**NICP** vertex displacement from bake is only for MP landmark transfer / debug meshes — **not** loaded in `ICTFaceKitTorch` or `ICTDeformer`.

## `ICTFaceKitTorch` runtime buffers (unified)

Npy keys are unchanged (`flame_alignment_*` / `flame_similarity_*`). At load, `_torch_pose_from_npy_dict` merges them into one rigid map:

| Buffer | Source |
|--------|--------|
| `flame_s`, `flame_R`, `flame_T` | `flame_alignment_*` if `flame_alignment_R` in npy, else `flame_similarity_s/T` + `R=I` |
| `use_flame_rigid` | `True` iff baked `flame_alignment_R` |
| `canonical_s`, `canonical_R`, `canonical_T` | optional `ict_identity.npy` (`canonical=` arg); default identity |

`use_flame_alignment` is a read-only alias of `use_flame_rigid`.

Two independent transforms:

1. **`apply_flame_similarity`**: `flame_s * (mesh @ flame_R) + flame_T` on the **FACS-deformed** mesh (default `True` in `forward`).
2. **`to_canonical_space`**: second stage `canonical_s * (mesh @ canonical_R) + canonical_T`.

Init:

- `self.expression[0, jaw_index] = flame_similarity_ict_jaw_open`
- `expression_canonical` = `forward(self.expression, apply_flame_similarity=True)` (jaw-open + FLAME map)
- `template_canonical` = same with `jawOpen=0`
- `canonical` = alias of `expression_canonical` (GS / scripts)
- `neutral_mesh_canonical` = copy of `expression_canonical`

See `docs/implementation/2026-05-29-dual-canonical.md`.

## `ICTDeformer`

- **`canonical_xyz_template`**: `ict.template_canonical[0]` — `template_mlp`, `pose_weight_net`.
- **`canonical_xyz_expression`**: `ict.expression_canonical[0]` — `expr_mlp` only.
- **Output mesh**: `ict.forward(..., apply_flame_similarity=True)` + deltas — FLAME-aligned ICT space, not NICP.

Do not use raw closed-mouth `neutral_mesh` for **expression** MLP input; lips collapse. Template MLP intentionally uses jaw-closed `template_canonical`.

## No double application of `s, R, T`

| Data in npy | Pre-transformed? |
|-------------|------------------|
| `neutral_mesh` | **No** — raw `generic_neutral_mesh.obj` vertices |
| `expression_shape_modes` | **No** — ICT-local deltas (same units as neutral) |
| `flame_alignment_*` / `flame_similarity_*` | Stored as scalars only, not baked into vertices |

Runtime: **one** `apply_flame_similarity` at the end of `ICTFaceKitTorch.forward` when `apply_flame_similarity=True`.

Legacy `to_canonical_space` (`canonical_s/R/T` from optional `ict_identity.npy`) is a **second**, independent pose — default identity when omitted. Do not pass `ict_identity.npy` together with full npy alignment unless you intend both.

NICP per-vertex displacement from bake is **not** loaded into the train stack.

## Blendshapes vs scale

FACS is linear in ICT space, then a **single** rigid similarity maps the whole mesh:

`align(neutral + Σ wᵢ modeᵢ) = align(neutral) + Σ wᵢ · s · (modeᵢ @ R)` (when `flame_alignment_*` is used).

Expression weights are dimensionless; they are **not** multiplied by `flame_alignment_s` separately — scale is inside `apply_flame_similarity`.

## `ICTDeformer` template / expr offsets

- MLP **input coords** split (template closed, expression jaw-open); added deltas still live in per-frame `forward` FLAME space.

## Sanity script

`debug/sanity_gaussian_layout.py` renders via `forward`, not raw `neutral_mesh`. Use `--compare-raw-neutral` for contrast.
