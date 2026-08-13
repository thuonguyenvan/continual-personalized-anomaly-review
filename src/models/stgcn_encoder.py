from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# NTU RGB+D 25-joint graph. Edges are written in the common ST-GCN convention.
_NTU_INWARD_1BASED = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
    (24, 25), (25, 12),
]


def _normalize_digraph(a: torch.Tensor) -> torch.Tensor:
    # Column-normalized directed adjacency, matching the usual ST-GCN graph normalization idea.
    degree = a.sum(dim=0)
    inv = torch.zeros_like(degree)
    nz = degree > 0
    inv[nz] = 1.0 / degree[nz]
    return a @ torch.diag(inv)


def ntu_adjacency(num_joints: int = 25) -> torch.Tensor:
    if num_joints != 25:
        raise ValueError("This pilot ST-GCN graph currently supports NTU's 25 joints only")
    self_a = torch.eye(num_joints, dtype=torch.float32)
    inward = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    outward = torch.zeros_like(inward)
    for i, j in _NTU_INWARD_1BASED:
        src, dst = i - 1, j - 1
        inward[dst, src] = 1.0
        outward[src, dst] = 1.0
    return torch.stack([self_a, _normalize_digraph(inward), _normalize_digraph(outward)], dim=0)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_subsets: int = 3):
        super().__init__()
        self.num_subsets = num_subsets
        self.proj = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: [N,C,T,V], a: [K,V,V]
        n, _, t, v = x.shape
        x = self.proj(x).view(n, self.num_subsets, -1, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, a)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        x = self.tcn(self.gcn(x, a)) + res
        return self.relu(x)


class STGCNEncoder(nn.Module):
    """Compact ST-GCN-style encoder for the NTU120 mechanism audit.

    Input: [B,T,V,3]. Output: L2-normalized embedding [B,D].

    This is intentionally a clean in-repo baseline rather than a claim of reproducing
    any particular published implementation bit-for-bit. It uses the NTU 25-joint graph,
    spatial graph convolution, temporal convolution, residual blocks, and global pooling.
    """

    def __init__(self, num_joints: int = 25, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.register_buffer("A", ntu_adjacency(num_joints), persistent=True)
        self.data_bn = nn.BatchNorm1d(num_joints * in_channels)
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, residual=False),
            STGCNBlock(64, 64),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, stride=2),
            STGCNBlock(128, 128),
            STGCNBlock(128, 256, stride=2),
            STGCNBlock(256, 256),
        ])
        self.embedding = nn.Linear(256, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,T,V,C], got {tuple(x.shape)}")
        n, t, v, c = x.shape
        if v != self.num_joints or c != self.in_channels:
            raise ValueError(f"Expected V={self.num_joints}, C={self.in_channels}; got V={v}, C={c}")

        # Batch-normalize the per-frame joint-coordinate vector, then restore graph layout.
        x = x.permute(0, 3, 2, 1).contiguous()  # [N,C,V,T]
        x = x.view(n, c * v, t)
        x = self.data_bn(x)
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()  # [N,C,T,V]

        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=(2, 3))
        z = self.norm(self.embedding(x))
        return F.normalize(z, p=2, dim=-1)
