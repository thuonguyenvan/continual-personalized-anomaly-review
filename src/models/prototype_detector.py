"""Minimal distance-based anomaly detector for gap validation.

This baseline is intentionally simple. It should establish whether the target
phenomenon exists before introducing a learned continual-personalization method.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PrototypeDetector:
    prototype: np.ndarray | None = None
    threshold: float | None = None

    def fit(self, embeddings: np.ndarray, threshold_quantile: float = 0.95) -> "PrototypeDetector":
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError("embeddings must have shape (N,D) with N>0")
        self.prototype = embeddings.mean(axis=0)
        scores = self.score_samples(embeddings)
        self.threshold = float(np.quantile(scores, threshold_quantile))
        return self

    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        if self.prototype is None:
            raise RuntimeError("Detector has not been fit")
        diff = embeddings - self.prototype[None, :]
        return np.linalg.norm(diff, axis=1)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Threshold has not been set")
        return (self.score_samples(embeddings) > self.threshold).astype(np.int64)

    def update_personal_prototype(self, personal_embeddings: np.ndarray, alpha: float = 0.5) -> None:
        """Naive baseline update; 0<alpha<=1 controls personal contribution."""
        if self.prototype is None:
            raise RuntimeError("Detector has not been fit")
        if len(personal_embeddings) == 0:
            return
        personal = personal_embeddings.mean(axis=0)
        self.prototype = (1.0 - alpha) * self.prototype + alpha * personal


def dual_prototype_scores(
    embeddings: np.ndarray,
    global_prototype: np.ndarray,
    personal_prototype: np.ndarray | None,
) -> np.ndarray:
    """Distance to the closest trusted normal prototype."""
    dg = np.linalg.norm(embeddings - global_prototype[None, :], axis=1)
    if personal_prototype is None:
        return dg
    dp = np.linalg.norm(embeddings - personal_prototype[None, :], axis=1)
    return np.minimum(dg, dp)
