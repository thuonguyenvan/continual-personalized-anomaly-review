from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def threshold_from_quantile(normal_scores: Iterable[float], q: float = 0.95) -> float:
    scores = np.asarray(list(normal_scores), dtype=np.float64)
    if scores.size == 0:
        raise ValueError("normal_scores must be non-empty")
    return float(np.quantile(scores, q))


def binary_metrics(
    normal_scores: Iterable[float],
    anomaly_scores: Iterable[float],
    threshold: float,
) -> Dict[str, float]:
    normal = np.asarray(list(normal_scores), dtype=np.float64)
    anomaly = np.asarray(list(anomaly_scores), dtype=np.float64)

    fp = int((normal > threshold).sum())
    tn = int((normal <= threshold).sum())
    tp = int((anomaly > threshold).sum())
    fn = int((anomaly <= threshold).sum())

    fpr = fp / max(fp + tn, 1)
    recall = tp / max(tp + fn, 1)
    fnr = fn / max(tp + fn, 1)

    return {
        "threshold": float(threshold),
        "fpr": float(fpr),
        "anomaly_recall": float(recall),
        "anomaly_fnr": float(fnr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def score_margin(
    personal_normal_scores: Iterable[float],
    protected_anomaly_scores: Iterable[float],
) -> Dict[str, float]:
    normal = np.asarray(list(personal_normal_scores), dtype=np.float64)
    anomaly = np.asarray(list(protected_anomaly_scores), dtype=np.float64)
    if normal.size == 0 or anomaly.size == 0:
        raise ValueError("Both score arrays must be non-empty")

    return {
        "mean_margin": float(anomaly.mean() - normal.mean()),
        "median_margin": float(np.median(anomaly) - np.median(normal)),
        "normal_mean": float(normal.mean()),
        "anomaly_mean": float(anomaly.mean()),
    }
