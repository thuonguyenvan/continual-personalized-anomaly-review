from __future__ import annotations

import numpy as np

from src.preprocessing.skeleton import select_primary_body


FEATURE_NAMES = [
    "root_net_disp",
    "root_max_disp",
    "root_vertical_range",
    "root_vertical_drop",
    "root_speed_mean",
    "root_speed_max",
    "root_accel_mean",
    "root_accel_max",
    "joint_speed_mean",
    "joint_speed_max",
    "joint_accel_mean",
    "joint_accel_max",
    "body_height_mean",
    "body_height_min",
    "body_height_drop",
    "torso_tilt_change",
]


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(x, axis=axis)


def extract_kinematic_features(raw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Extract deterministic motion/pose summaries from an NTU skeleton sequence.

    The features intentionally retain world-coordinate root motion. They are used only
    for a diagnostic one-class audit, not as a proposed model.
    """
    x = select_primary_body(raw).astype(np.float32, copy=False)  # [T,V,3]
    if x.ndim != 3 or x.shape[1] != 25 or x.shape[2] != 3:
        raise ValueError(f"Expected primary body [T,25,3], got {x.shape}")
    if x.shape[0] < 2:
        x = np.concatenate([x, x], axis=0)

    root = x[:, 0, :]
    root_rel = root - root[:1]
    root_disp = _safe_norm(root_rel)
    root_vel = np.diff(root, axis=0)
    root_speed = _safe_norm(root_vel)
    root_acc = np.diff(root_vel, axis=0) if len(root_vel) > 1 else np.zeros((1, 3), dtype=np.float32)
    root_accel = _safe_norm(root_acc)

    joint_vel = np.diff(x, axis=0)
    joint_speed = _safe_norm(joint_vel)
    joint_acc = np.diff(joint_vel, axis=0) if len(joint_vel) > 1 else np.zeros((1, 25, 3), dtype=np.float32)
    joint_accel = _safe_norm(joint_acc)

    y = x[:, :, 1]
    body_height = y.max(axis=1) - y.min(axis=1)

    # NTU joints: 1=spine base (idx 0), 21=spine shoulder (idx 20).
    torso = x[:, 20, :] - x[:, 0, :]
    torso_len = np.maximum(_safe_norm(torso), eps)
    # Angle to the global vertical axis. Absolute cosine reduces sign sensitivity.
    cos_vertical = np.clip(np.abs(torso[:, 1]) / torso_len, 0.0, 1.0)
    torso_tilt = np.arccos(cos_vertical)

    feats = np.asarray([
        float(_safe_norm(root[-1] - root[0])),
        float(root_disp.max()),
        float(root[:, 1].max() - root[:, 1].min()),
        float(max(0.0, root[0, 1] - root[:, 1].min())),
        float(root_speed.mean()),
        float(root_speed.max()),
        float(root_accel.mean()),
        float(root_accel.max()),
        float(joint_speed.mean()),
        float(joint_speed.max()),
        float(joint_accel.mean()),
        float(joint_accel.max()),
        float(body_height.mean()),
        float(body_height.min()),
        float(max(0.0, body_height[0] - body_height.min())),
        float(torso_tilt.max() - torso_tilt.min()),
    ], dtype=np.float32)
    if not np.isfinite(feats).all():
        raise ValueError("Non-finite kinematic feature encountered")
    return feats
