"""Per-vertex pose weight w(x) in [0, 1] for ICT mesh."""

import torch
import torch.nn as nn


class PoseWeightMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, canonical_xyz):
        """
        canonical_xyz: [V, 3] or [B, V, 3]
        returns: same batch dims, [..., 1] in (0, 1)
        """
        w = self.mlp(canonical_xyz)
        return torch.sigmoid(w)
