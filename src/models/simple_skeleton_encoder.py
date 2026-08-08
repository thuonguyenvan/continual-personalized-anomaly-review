from __future__ import annotations

import torch
from torch import nn


class SimpleSkeletonEncoder(nn.Module):
    """Small temporal encoder for pilot experiments.

    Input shape: [B, T, V, C], with C=3 xyz coordinates.
    Output: L2-normalized embedding [B, D].

    This model is intentionally simple. The first objective is to test whether
    subject-specific/new-normal distribution shift exists before introducing a
    stronger backbone.
    """

    def __init__(self, num_joints: int = 25, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        feature_dim = num_joints * in_channels
        self.frame_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )
        self.temporal = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,V,C], got {tuple(x.shape)}")
        b, t, v, c = x.shape
        x = x.reshape(b, t, v * c)
        x = self.frame_proj(x)
        _, h = self.temporal(x)
        z = self.norm(h[-1])
        return torch.nn.functional.normalize(z, p=2, dim=-1)
