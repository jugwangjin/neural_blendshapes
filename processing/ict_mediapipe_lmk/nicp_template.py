"""NICP bake template: extension-region displacement + merge into ``ict_facekit_torch.npy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from processing.ict_mediapipe_lmk.constants import ICT_FACE_VERTEX_END


def nicp_extension_vertex_indices(regions: dict) -> np.ndarray:
    """
    Vertices outside the skin-face NICP patch that should follow face NICP
    (mouth socket, eye sockets, mouth interior, eye occlusion).
    Eyeballs / teeth / eyelashes are excluded.
    """
    keys = (
        "mouth_socket_indices",
        "eye_socket_left_indices",
        "eye_socket_right_indices",
        "gums_tongue_indices",
        "mouth_interior_vertex_indices",
        "left_eye_occlusion_indices",
        "right_eye_occlusion_indices",
    )
    out = []
    for k in keys:
        if k in regions and regions[k]:
            out.extend(regions[k])
    return np.asarray(sorted(set(int(i) for i in out)), dtype=np.int64)


def build_vertex_neighbors(faces: np.ndarray, n_verts: int) -> list[list[int]]:
    faces = np.asarray(faces, dtype=np.int64)
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    for tri in faces.reshape(-1, 3):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for i, j in ((a, b), (b, c), (c, a)):
            if i < n_verts and j < n_verts:
                nbrs[i].add(j)
                nbrs[j].add(i)
    return [sorted(s) for s in nbrs]


def propagate_nicp_displacement(
    v_ref: np.ndarray,
    v_after_face_nicp: np.ndarray,
    faces: np.ndarray,
    extension_vids: np.ndarray,
    *,
    face_end: int = ICT_FACE_VERTEX_END,
    n_iters: int = 12,
) -> np.ndarray:
    """
    Copy face NICP (``0:face_end``) then diffuse displacement to ``extension_vids``
    from mesh neighbors (face patch + extension ring).
    """
    v_ref = np.asarray(v_ref, dtype=np.float64)
    v_out = np.asarray(v_after_face_nicp, dtype=np.float64).copy()
    n_v = v_ref.shape[0]
    ext = [int(v) for v in np.asarray(extension_vids, dtype=np.int64).ravel() if face_end <= int(v) < n_v]
    if not ext:
        return v_out

    nbrs = build_vertex_neighbors(faces, n_v)
    ext_set = set(ext)
    for _ in range(n_iters):
        for vid in ext:
            nb = [u for u in nbrs[vid] if u < face_end or u in ext_set]
            if not nb:
                continue
            disp = np.mean([v_out[u] - v_ref[u] for u in nb], axis=0)
            v_out[vid] = v_ref[vid] + disp
    return v_out


def apply_nicp_extension_to_full_mesh(
    v_ref: np.ndarray,
    v_ict_fit: np.ndarray,
    faces: np.ndarray,
    regions: dict,
    *,
    n_iters: int = 12,
) -> np.ndarray:
    ext = nicp_extension_vertex_indices(regions)
    return propagate_nicp_displacement(
        v_ref, v_ict_fit, faces, ext, n_iters=n_iters
    )


def merge_nicp_canonical_into_npy(
    npy_path: str | Path,
    nicp_canonical_mesh: np.ndarray,
    *,
    jaw_open: float,
    bake_script: str = "bake_mediapipe_to_ict.py",
) -> Path:
    """Write ``nicp_canonical_mesh`` into ``ict_facekit_torch.npy`` (in-place update)."""
    npy_path = Path(npy_path)
    d = np.load(str(npy_path), allow_pickle=True).item()
    v = np.asarray(nicp_canonical_mesh, dtype=np.float64)
    neutral = np.asarray(d["neutral_mesh"], dtype=np.float64)
    if v.shape != neutral.shape:
        raise ValueError(f"nicp_canonical_mesh {v.shape} != neutral_mesh {neutral.shape}")
    d["nicp_canonical_mesh"] = v.astype(np.float32)
    d["nicp_bake_jaw_open"] = float(jaw_open)
    d["nicp_bake_source"] = bake_script
    np.save(str(npy_path), d)
    return npy_path
