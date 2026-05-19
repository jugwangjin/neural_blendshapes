# ICT → FLAME initial alignment (npy + MediaPipe bake)

## Root cause of “garbage initial”

`optimize_ict_expression_to_flame.py` aligns with **pytorch3d `s,R,T`** on landmarks `[17:]`.  
Using **uniform `s,T` only** leaves a large rotational mismatch → initial debug looks wrong even when scalar landmark error looks small.

## Convention (current)

| Side | Mesh / landmarks |
|------|------------------|
| FLAME | `exp=0`, `full_pose=canonical_pose`; 68 bary landmarks from `flame_static_embedding.pkl` |
| ICT | `neutral_mesh + jawOpen * mode`; vertex indices `landmark_indices[17:]` (Multi-PIE jawline 68) |

Transform stored in npy (default):

```text
x_flame = flame_alignment_s * (x_ict_jaw @ flame_alignment_R) + flame_alignment_T
```

`flame_similarity_s/T` are identity (1, 0) — the real fit is in `flame_alignment_*`.

Legacy `--coarse_st_only`: jaw grid + uniform `flame_similarity_s/T` only (no rotation).

## Code

- `processing/ict_flame_similarity.py` — `optimize_ict_jaw_open(..., use_rigid=True)`, `fit_rigid_alignment_fields`, `apply_ict_to_flame_space`
- `processing/ict_facekit_to_npy_full_head.py` — writes alignment into `assets/ict_facekit_torch.npy`
- `processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py` — loads `neutral_mesh`, applies npy transform, exports `ict_initial_flame_space.obj`

Bake uses **`flame_alignment_*` automatically** when present in npy.  
If you have an old npy (coarse only or bad composed `flame_alignment`), **rebuild npy** then bake.

## Bake (`bake_mediapipe_to_ict.py`)

- Loads ICT **neutral** from npy, applies same `apply_ict_to_flame_space` as npy (`flame_alignment_*` when present).
- FLAME canonical mesh uses **npy flags** (`flame_similarity_use_processed_faces`, `use_canonical_pose`) — not ad-hoc CLI mismatch.
- Face NICP uses `pair_landmarks_for_alignment` (same 51/68 rule); old code used `len(flame)-17` → 34 pairs bug.
- Eye transplant: `v_ict` (pre-NICP, aligned); face MP transfer: `v_ict_fit`.

## Regenerate

```bash
python processing/ict_facekit_to_npy_full_head.py
python processing/ict_mediapipe_lmk/bake_mediapipe_to_ict.py
```

Check logs:

- `FLAME landmarks: 51 pts, paired n=51` — inner embedding (do **not** slice `[17:]` again)
- `FLAME landmarks: 68 pts, paired n=51` — full Multi-PIE, slice `[17:]`
- npy build: `after jaw+s,R,T: mean=...` should be small (mm scale, e.g. &lt; 0.005 m). **~0.17 m means only 34 pairs were used (bug: 51-pt FLAME sliced at 17).**
- bake: same `print_flame_alignment_report` before NICP

Debug meshes: `processing/ict_mediapipe_lmk/debug/ict_initial_flame_space.obj` vs `flame_canonical.obj`, landmark PLYs.

## Stale npy

Old bake may have written **composed** `flame_alignment` (coarse s,T + second rigid). Re-run full_head; optional bake `--recompute_flame_alignment` if npy lacks `flame_alignment_*`.
