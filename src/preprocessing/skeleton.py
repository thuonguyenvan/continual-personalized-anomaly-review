"""Skeleton preprocessing for the NTU120 personalization pilot."""

from __future__ import annotations

import numpy as np


def select_primary_body(x: np.ndarray) -> np.ndarray:
    """Select the body with the largest accumulated motion/visibility."""
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
    """Subtract the root joint independently at every frame.

    This removes global translation. It remains the legacy/default preprocessing
    because earlier pilot checkpoints were trained with it.
    """
    root = x[:, root_joint : root_joint + 1, :]
    return x - root


def sequence_origin_center(x: np.ndarray, root_joint: int = 0) -> np.ndarray:
    """Subtract only the first-frame root position.

    Absolute camera location is removed while the subsequent root trajectory is
    retained. This is useful for safety-motion audits because frame-wise root
    centering can erase vertical/horizontal body displacement during a fall.
    """
    origin = x[0:1, root_joint : root_joint + 1, :]
    return x - origin


def robust_body_scale_from_pose(x: np.ndarray, root_joint: int = 0, eps: float = 1e-6) -> float:
    """Estimate body scale from root-relative pose, excluding global trajectory."""
    pose = root_center(x, root_joint=root_joint)
    d = np.linalg.norm(pose, axis=-1)
    valid = d[d > eps]
    if valid.size == 0:
        return 1.0
    return max(float(np.median(valid)), eps)


def normalize_body_scale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize an already-centered sequence by its median non-zero radius."""
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


def preprocess_skeleton(x: np.ndarray, target_len: int = 64, mode: str = "frame_root") -> np.ndarray:
    """Preprocess an NTU skeleton sequence.

    Modes
    -----
    frame_root:
        Legacy pilot mode: primary body -> per-frame root center -> scale -> resample.
        Removes global body translation.
    sequence_origin:
        Primary body -> subtract first-frame root only -> scale using root-relative
        body size -> resample. Retains root trajectory while removing absolute
        camera-space location.
    """
    x = select_primary_body(x)
    if mode == "frame_root":
        x = root_center(x)
        x = normalize_body_scale(x)
    elif mode == "sequence_origin":
        scale = robust_body_scale_from_pose(x)
        x = sequence_origin_center(x) / scale
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode!r}")
    x = temporal_resample(x, target_len=target_len)
    return x.astype(np.float32, copy=False)
