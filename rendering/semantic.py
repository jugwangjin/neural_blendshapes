"""Semantic class ids and per-class h priors for avatar Gaussians."""

import torch

SEMANTIC_CLASSES = (
    "skin",
    "lip",
    "eye",
    "iris",
    "hair",
    "accessory",
    "bg",
)

SEMANTIC_CLASS_INDEX = {name: i for i, name in enumerate(SEMANTIC_CLASSES)}

H_PRIOR = {
    "skin": {"sigma": 0.002, "weight": 1.0},
    "lip": {"sigma": 0.003, "weight": 0.8},
    "eye": {"sigma": 0.002, "weight": 1.0},
    "iris": {"sigma": 0.002, "weight": 1.0},
    "hair": {"sigma": 0.030, "weight": 0.1},
    "accessory": {"sigma": 0.100, "weight": 0.0},
    "bg": {"sigma": 0.100, "weight": 0.0},
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
