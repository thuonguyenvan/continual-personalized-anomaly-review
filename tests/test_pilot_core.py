import numpy as np

from src.models.prototype_detector import PrototypeDetector, dual_prototype_scores
from src.preprocessing.skeleton import preprocess_skeleton, temporal_resample


def test_temporal_resample_shape_and_dtype():
    x = np.random.default_rng(0).normal(size=(10, 25, 3)).astype(np.float32)
    y = temporal_resample(x, target_len=64)
    assert y.shape == (64, 25, 3)
    assert y.dtype == np.float32


def test_preprocess_skeleton_shape_and_root_centering():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 2, 25, 3)).astype(np.float32)
    y = preprocess_skeleton(x, target_len=32)
    assert y.shape == (32, 25, 3)
    assert np.allclose(y[:, 0, :], 0.0, atol=1e-5)


def test_sequence_origin_preprocessing_preserves_root_trajectory():
    x = np.zeros((8, 1, 25, 3), dtype=np.float32)
    base = np.linspace(0.0, 1.0, 25, dtype=np.float32)
    for t in range(8):
        x[t, 0, :, 0] = base
        x[t, 0, :, 1] = base * 0.5 - 0.1 * t
    y = preprocess_skeleton(x, target_len=8, mode="sequence_origin")
    assert y.shape == (8, 25, 3)
    assert np.allclose(y[0, 0], 0.0, atol=1e-6)
    assert y[-1, 0, 1] < -0.5


def test_prototype_detector_orders_near_and_far_samples():
    train = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    det = PrototypeDetector().fit(train, threshold_quantile=0.95)
    scores = det.score_samples(np.asarray([[0.05, 0.05], [3.0, 3.0]], dtype=np.float32))
    assert scores[0] < scores[1]


def test_dual_prototype_can_reduce_personal_distance_without_moving_global():
    x = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    g = np.asarray([0.0, 0.0], dtype=np.float32)
    p = np.asarray([10.0, 10.0], dtype=np.float32)
    single = dual_prototype_scores(x, g, None)
    dual = dual_prototype_scores(x, g, p)
    assert np.isclose(single[0], dual[0])
    assert dual[1] < single[1]
