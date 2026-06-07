"""Semantic class ids for mesh-embedded Gaussian seg render (3-class + ignore)."""

import torch

# ICT surface + FLARE seg: only classes both can assign reliably.
# ``others`` = face/head/lips/ears/…; mouth/eye sockets are in mouth_interior / eye_occlusion.
SEMANTIC_CLASSES = (
    "others",
    "mouth_interior",
    "eye_occlusion",
)

SEMANTIC_CLASS_INDEX = {name: i for i, name in enumerate(SEMANTIC_CLASSES)}

SEMANTIC_IGNORE_INDEX = -1

H_PRIOR = {
    "others": {"sigma": 0.002, "weight": 1.0},
    "mouth_interior": {"sigma": 0.002, "weight": 1.0},
    "eye_occlusion": {"sigma": 0.002, "weight": 1.0},
}


def h_prior_tensors(device, dtype=torch.float32):
    sigma = torch.tensor(
        [H_PRIOR[c]["sigma"] for c in SEMANTIC_CLASSES], device=device, dtype=dtype
    )
    weight = torch.tensor(
        [H_PRIOR[c]["weight"] for c in SEMANTIC_CLASSES], device=device, dtype=dtype
    )
    return sigma, weight


DEFAULT_H_SIGMA = tuple(H_PRIOR[c]["sigma"] for c in SEMANTIC_CLASSES)
