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

## `ICTFaceKitTorch` usage

Two independent transforms:

1. **`apply_flame_similarity`** (default `True` in `forward`): uses npy `flame_alignment_*` if present, else `flame_similarity_s/T`. Applied to the **FACS-deformed** mesh (expression weights from caller).
2. **`to_canonical_space`**: legacy `s,R,T` from optional `ict_identity.npy` (`canonical=` arg). Default when omitted: identity (`s=1`, `R=I`, `T=0`).

Init:

- `self.expression[0, jaw_index] = flame_similarity_ict_jaw_open`
- `self.canonical` = `forward(default expression, to_canonical=True)` → jaw-open + FLAME alignment (+ legacy pose if provided)
- `neutral_mesh_canonical` = same as `canonical` (jaw-open reference for normals / MLP coords)

## `ICTDeformer`

- **`canonical_xyz`**: `ict.canonical[0]` (jaw-open, same space as `forward(..., apply_flame_similarity=True)`).
- **Output mesh**: `ict.forward(..., to_canonical=False, apply_flame_similarity=True)` on template + MP expression — FLAME-aligned ICT space, not NICP.

Do not use closed-mouth `neutral_mesh` alone as MLP input; lips collapse.

## No double application of `s, R, T`

| Data in npy | Pre-transformed? |
|-------------|------------------|
| `neutral_mesh` | **No** — raw `generic_neutral_mesh.obj` vertices |
| `expression_shape_modes` | **No** — ICT-local deltas (same units as neutral) |
| `flame_alignment_*` / `flame_similarity_*` | Stored as scalars only, not baked into vertices |

Runtime: **one** `apply_flame_similarity` at the end of `ICTFaceKitTorch.forward` when `apply_flame_similarity=True`.

Legacy `to_canonical_space` (`s,R,T` from optional `ict_identity.npy`) is a **second**, independent pose — default identity when omitted. Do not pass `ict_identity.npy` together with full npy alignment unless you intend both.

NICP per-vertex displacement from bake is **not** loaded into the train stack.

## Blendshapes vs scale

FACS is linear in ICT space, then a **single** rigid similarity maps the whole mesh:

`align(neutral + Σ wᵢ modeᵢ) = align(neutral) + Σ wᵢ · s · (modeᵢ @ R)` (when `flame_alignment_*` is used).

Expression weights are dimensionless; they are **not** multiplied by `flame_alignment_s` separately — scale is inside `apply_flame_similarity`.

## `ICTDeformer` template / expr offsets

- `canonical_xyz` = `ict.canonical[0]` (jaw-open + aligned).
- `forward` → aligned mesh; `template_delta` and `expression_delta` are added **in that same space** (not via raw `neutral_mesh` swap).

## Sanity script

`scripts/sanity_gaussian_layout.py` renders via `forward`, not raw `neutral_mesh`. Use `--compare-raw-neutral` to see misalignment from skipping FACS/alignment. Use `--sweep-jaw` / `--expr` for visual FACS checks.
