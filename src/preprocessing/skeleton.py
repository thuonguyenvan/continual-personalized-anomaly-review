"""Skeleton preprocessing for the first NTU120 personalization pilot."""

from __future__ import annotations

import numpy as np


def select_primary_body(x: np.ndarray) -> np.ndarray:
    """Select the body with the largest accumulated motion/visibility.

    Parameters
    ----------
    x : np.ndarray
        Shape (T, M, V, 3).

    Returns
    -------
    np.ndarray
        Shape (T, V, 3).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (T,M,V,3), got {x.shape}")
    visibility = (np.abs(x).sum(axis=-1) > 0).sum(axis=(0, 2))
    if x.shape[0] > 1:
        motion = np.abs(np.diff(x, axis=0)).sum(axis=(0, 2, 3))
    else:
        motion = np.zeros(x.shape[1], dtype=np.float32)
    score = visibility.astype(np.float32) + motion
    return x[:, int(np.argmax(score))]


def root_center(x: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """Subtract the root joint coordinates frame-wise."""
    root = x[:, root_joint : root_joint + 1, :]
    return x - root


def normalize_body_scale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize by a robust per-sequence body scale.

    Uses the median non-zero distance of joints from the root-centered origin.
    This intentionally avoids assuming a particular bone indexing convention.
    """
    d = np.linalg.norm(x, axis=-1)
    valid = d[d > eps]
    if valid.size == 0:
        return x
    scale = float(np.median(valid))
    return x / max(scale, eps)


def temporal_resample(x: np.ndarray, target_len: int = 64) -> np.ndarray:
    """Linearly resample sequence along time to a fixed length."""
    if x.shape[0] == 0:
        raise ValueError("Cannot resample an empty sequence")
    if x.shape[0] == target_len:
        return x.astype(np.float32, copy=False)

    src = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    flat = x.reshape(x.shape[0], -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape(target_len, *x.shape[1:])


def preprocess_skeleton(x: np.ndarray, target_len: int = 64) -> np.ndarray:
    """Canonical pilot preprocessing: primary body -> center -> scale -> resample."""
    x = select_primary_body(x)
    x = root_center(x)
    x = normalize_body_scale(x)
    x = temporal_resample(x, target_len=target_len)
    return x.astype(np.float32, copy=False)
