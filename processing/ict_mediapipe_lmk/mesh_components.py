"""Mesh connected components and submesh extraction for eye-only NICP."""

import numpy as np


def build_vertex_adjacency(faces, n_vertices):
    adj = [[] for _ in range(n_vertices)]
    for tri in faces:
        a, b, c = map(int, tri)
        adj[a].extend([b, c])
        adj[b].extend([a, c])
        adj[c].extend([a, b])
    return [np.unique(x).astype(np.int64) for x in adj]


def connected_component_from_seeds(faces, n_vertices, seeds):
    """Connected vertex component containing all seed vertices."""
    adj = build_vertex_adjacency(faces, n_vertices)
    seeds = np.asarray(seeds, dtype=np.int64)

    visited = np.zeros(n_vertices, dtype=bool)
    stack = [int(seeds[0])]
    visited[seeds[0]] = True

    while stack:
        v = stack.pop()
        for nb in adj[v]:
            nb = int(nb)
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)

    if not np.all(visited[seeds]):
        bad = seeds[~visited[seeds]]
        raise RuntimeError(f"Iris seeds not in one connected component: {bad.tolist()}")

    return np.where(visited)[0].astype(np.int64)


def extract_submesh(vertices, faces, vertex_ids):
    """
    Returns sub_vertices, sub_faces, global_face_ids, global_to_local dict.
    """
    vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
    keep_v = np.zeros(len(vertices), dtype=bool)
    keep_v[vertex_ids] = True

    face_mask = np.all(keep_v[faces], axis=1)
    global_face_ids = np.where(face_mask)[0]
    sub_faces_global = faces[face_mask]

    global_to_local = {int(g): i for i, g in enumerate(vertex_ids)}
    sub_faces = np.array(
        [[global_to_local[int(v)] for v in tri] for tri in sub_faces_global],
        dtype=np.int64,
    )
    sub_vertices = vertices[vertex_ids]
    return sub_vertices, sub_faces, global_face_ids, global_to_local


def local_indices(global_ids, global_to_local):
    return np.array([global_to_local[int(v)] for v in global_ids], dtype=np.int64)


def extract_flame_eye_components(v_flame, f_flame, left_iris_seeds, right_iris_seeds):
    n = len(v_flame)
    left_eye = connected_component_from_seeds(f_flame, n, left_iris_seeds)
    right_eye = connected_component_from_seeds(f_flame, n, right_iris_seeds)

    inter = np.intersect1d(left_eye, right_eye)
    if inter.size > 0:
        raise RuntimeError(f"FLAME left/right eye components overlap: {inter.size} verts")

    if len(left_eye) > 0.2 * n or len(right_eye) > 0.2 * n:
        print(
            "[Warning] FLAME eye component is large (>20% of mesh). "
            "Eyeball may not be a disconnected component on this FLAME mesh."
        )

    print("[FLAME eye components]")
    print(f"  left eyeball: {len(left_eye)} verts")
    print(f"  right eyeball: {len(right_eye)} verts")
    return left_eye, right_eye


def load_ict_eye_regions_from_dict(model_dict):
    v_ict = np.asarray(model_dict["neutral_mesh"], dtype=np.float64)
    f_ict = np.asarray(model_dict["faces"], dtype=np.int64)

    left_eye = np.asarray(
        model_dict.get("left_eyeball_indices", np.arange(21451, 23021)),
        dtype=np.int64,
    )
    right_eye = np.asarray(
        model_dict.get("right_eyeball_indices", np.arange(23021, 24591)),
        dtype=np.int64,
    )
    left_iris = np.asarray(model_dict.get("left_iris_indices", []), dtype=np.int64)
    right_iris = np.asarray(model_dict.get("right_iris_indices", []), dtype=np.int64)

    print("[ICT eye regions]")
    print(f"  V={v_ict.shape[0]} left_eyeball={len(left_eye)} right_eyeball={len(right_eye)}")
    print(f"  left_iris={len(left_iris)} right_iris={len(right_iris)}")
    return v_ict, f_ict, left_eye, right_eye, left_iris, right_iris


def check_projected_faces_are_in_eye(face_idx, f_ict, eye_vertex_ids, side="eye"):
    eye_set = set(map(int, eye_vertex_ids))
    for fid in face_idx:
        tri = f_ict[int(fid)]
        if not all(int(v) in eye_set for v in tri):
            raise RuntimeError(
                f"{side}: projected landmark not on eye component: face {fid}, tri={tri.tolist()}"
            )
