"""Load assets/ict_facekit_torch.npy without importing model package (bake / tooling)."""

import numpy as np

from processing.ict_region_dict import build_official_region_indices


def load_ict_npy_dict(npy_path):
    return np.load(npy_path, allow_pickle=True).item()


def regions_from_npy_dict(model_dict):
    official = build_official_region_indices()
    keys = [
        "face_indices",
        "not_face_indices",
        "eyeball_indices",
        "head_indices",
        "left_eyeball_indices",
        "right_eyeball_indices",
        "left_iris_indices",
        "right_iris_indices",
        "face_material_name",
        "triangle_uv_local",
        "eye_socket_left_indices",
        "eye_socket_right_indices",
        "surface_sample_vertex_indices",
        "skin_face_indices",
        "head_neck_indices",
        "mouth_socket_indices",
        "mouth_interior_vertex_indices",
        "gums_tongue_indices",
        "teeth_indices",
    ]
    regions = {}
    for k in keys:
        if k in model_dict:
            regions[k] = model_dict[k]
        elif k in official:
            regions[k] = official[k]
    if "gums_tongue_indices" not in regions and "mouth_interior_vertex_indices" in regions:
        regions["gums_tongue_indices"] = regions["mouth_interior_vertex_indices"]
    regions["asset_variant"] = model_dict.get("asset_variant", "unknown")
    regions["asset_schema_version"] = int(model_dict.get("asset_schema_version", 0))
    return regions


def load_ict_asset(npy_path, apply_flame_similarity_transform=True):
    """
    Returns v_ict, f_ict, uvs, uv_faces, landmark_indices (legacy 68), regions, raw dict.

    If ``apply_flame_similarity_transform``: ``neutral + jawOpen`` then FLAME space
    (``flame_alignment_s,R,T`` when baked, else coarse ``flame_similarity_s,T``).
    """
    from processing.ict_flame_similarity import apply_ict_to_flame_space

    d = load_ict_npy_dict(npy_path)
    v_ict = np.asarray(d["neutral_mesh"], dtype=np.float64)
    if apply_flame_similarity_transform and "flame_similarity_s" in d:
        from processing.ict_flame_similarity import has_flame_alignment

        v_ict = apply_ict_to_flame_space(
            v_ict, d, use_final_alignment=has_flame_alignment(d)
        )
    f_ict = np.asarray(d["faces"], dtype=np.int64)
    uvs = np.asarray(d["uvs"], dtype=np.float64)
    uv_faces = np.asarray(d["uv_faces"], dtype=np.int64)
    lmk = d.get("landmark_indices", [])
    regions = regions_from_npy_dict(d)
    return v_ict, f_ict, uvs, uv_faces, lmk, regions, d
