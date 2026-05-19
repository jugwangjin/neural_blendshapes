"""Training-time helpers for ICT MediaPipe embedding (baked under assets/)."""

import numpy as np
import torch

from processing.ict_mediapipe_lmk.landmarks import vertices2landmarks


def load_ict_mediapipe_embedding(path):
    return np.load(path, allow_pickle=True)


def ict_vertices_to_mediapipe(vertices, faces, embedding_npz):
    """
    vertices: [B, V, 3]
    faces: [F, 3]
  embedding_npz: loaded np.load(...) object
    returns: [B, L, 3]
    """
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces, dtype=torch.long, device=vertices.device)
    return vertices2landmarks(
        vertices,
        faces,
        torch.tensor(embedding_npz["ict_lmk_face_idx"], dtype=torch.long, device=vertices.device),
        torch.tensor(embedding_npz["ict_lmk_b_coords"], dtype=torch.float32, device=vertices.device),
    )
