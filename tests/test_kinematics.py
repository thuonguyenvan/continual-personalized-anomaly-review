import numpy as np

from src.features.kinematics import FEATURE_NAMES, extract_kinematic_features


def test_kinematic_feature_shape_and_finite():
    x = np.zeros((12, 2, 25, 3), dtype=np.float32)
    for t in range(12):
        x[t, 0, :, 1] = np.linspace(0.0, 1.0, 25) - 0.05 * t
        x[t, 0, :, 0] = np.linspace(0.0, 0.2, 25)
    f = extract_kinematic_features(x)
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_vertical_drop_increases_for_falling_root():
    standing = np.zeros((10, 1, 25, 3), dtype=np.float32)
    falling = np.zeros_like(standing)
    for t in range(10):
        standing[t, 0, :, 1] = np.linspace(0.0, 1.0, 25)
        falling[t, 0, :, 1] = np.linspace(0.0, 1.0, 25) - 0.1 * t
    fs = extract_kinematic_features(standing)
    ff = extract_kinematic_features(falling)
    idx = FEATURE_NAMES.index("root_vertical_drop")
    assert ff[idx] > fs[idx]
