from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from scripts.run_personalization_baselines import load_embeddings, mask_rows


def quantile_threshold(scores: np.ndarray, q: float) -> float:
    if len(scores) == 0:
        raise ValueError("cannot calibrate threshold on empty scores")
    return float(np.quantile(scores, q))


def evaluate(name: str, scores_calib: np.ndarray, scores_retention: np.ndarray,
             scores_personal: np.ndarray, scores_safe: np.ndarray, q: float) -> Dict[str, float | str]:
    threshold = quantile_threshold(scores_calib, q)
    personal_fpr = float(np.mean(scores_personal > threshold))
    safe_recall = float(np.mean(scores_safe > threshold))
    retention_fpr = float(np.mean(scores_retention > threshold))
    p_mean = float(np.mean(scores_personal))
    s_mean = float(np.mean(scores_safe))
    return {
        "detector": name,
        "threshold": threshold,
        "retention_fpr": retention_fpr,
        "personal_fpr": personal_fpr,
        "safe_recall": safe_recall,
        "safe_fnr": 1.0 - safe_recall,
        "personal_score_mean": p_mean,
        "safe_score_mean": s_mean,
        "score_margin": s_mean - p_mean,
    }


def centroid_scores(z_train: np.ndarray, z: np.ndarray) -> np.ndarray:
    p = z_train.mean(axis=0)
    return np.linalg.norm(z - p[None, :], axis=1)


def kmeans_scores(z_train: np.ndarray, z: np.ndarray, k: int, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    model.fit(z_train)
    centers = model.cluster_centers_
    d = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    return d.min(axis=1)


def knn_fit(z_train: np.ndarray, k: int) -> NearestNeighbors:
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(z_train)
    return model


def knn_scores(model: NearestNeighbors, z: np.ndarray) -> np.ndarray:
    d, _ = model.kneighbors(z)
    return d.mean(axis=1)


def ocsvm_fit(z_train: np.ndarray, nu: float, gamma: str) -> OneClassSVM:
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(z_train)
    return model


def ocsvm_scores(model: OneClassSVM, z: np.ndarray) -> np.ndarray:
    return -model.decision_function(z).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit frozen Session-0 detector geometry without using deployment anomalies for tuning"
    )
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/session0/detector_audit.csv")
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--kmeans-k", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--knn-k", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--ocsvm-nu", type=float, default=0.05)
    ap.add_argument("--ocsvm-gamma", default="scale")
    ap.add_argument("--ocsvm-max-train", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tab = load_embeddings(args.embeddings, args.metadata)
    z, rows = tab.z, tab.rows

    m_train = mask_rows(rows, inner_split="encoder_train", role="global_normal")
    m_calib = mask_rows(rows, inner_split="detector_calib", role="global_normal")
    m_ret = mask_rows(rows, inner_split="retention_val", role="global_normal")
    m_personal = mask_rows(rows, inner_split="deployment_test", role="candidate_personal_normal")
    m_safe = mask_rows(rows, inner_split="deployment_test", role="protected_anomaly")

    z_train = z[m_train]
    z_calib = z[m_calib]
    z_ret = z[m_ret]
    z_personal = z[m_personal]
    z_safe = z[m_safe]

    if min(len(z_train), len(z_calib), len(z_ret), len(z_personal), len(z_safe)) == 0:
        raise RuntimeError("one or more required partitions are empty")

    results: List[Dict[str, float | str]] = []

    results.append(evaluate(
        "centroid",
        centroid_scores(z_train, z_calib),
        centroid_scores(z_train, z_ret),
        centroid_scores(z_train, z_personal),
        centroid_scores(z_train, z_safe),
        args.threshold_quantile,
    ))

    for k in args.kmeans_k:
        if k <= 0 or k > len(z_train):
            continue
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit(z_train)
        def score_km(x: np.ndarray) -> np.ndarray:
            d = np.linalg.norm(x[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
            return d.min(axis=1)
        results.append(evaluate(
            f"kmeans_{k}", score_km(z_calib), score_km(z_ret),
            score_km(z_personal), score_km(z_safe), args.threshold_quantile,
        ))

    for k in args.knn_k:
        if k <= 0 or k > len(z_train):
            continue
        nn = knn_fit(z_train, k)
        results.append(evaluate(
            f"knn_{k}", knn_scores(nn, z_calib), knn_scores(nn, z_ret),
            knn_scores(nn, z_personal), knn_scores(nn, z_safe), args.threshold_quantile,
        ))

    if len(z_train) > args.ocsvm_max_train:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(z_train), size=args.ocsvm_max_train, replace=False)
        z_oc = z_train[keep]
    else:
        z_oc = z_train
    oc = ocsvm_fit(z_oc, args.ocsvm_nu, args.ocsvm_gamma)
    results.append(evaluate(
        "ocsvm", ocsvm_scores(oc, z_calib), ocsvm_scores(oc, z_ret),
        ocsvm_scores(oc, z_personal), ocsvm_scores(oc, z_safe), args.threshold_quantile,
    ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["detector", "threshold", "retention_fpr", "personal_fpr", "safe_recall", "safe_fnr",
              "personal_score_mean", "safe_score_mean", "score_margin"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"train_global_normal: {len(z_train)}")
    print(f"calib_global_normal: {len(z_calib)}")
    print(f"retention_global_normal: {len(z_ret)}")
    print(f"deployment_personal_normal: {len(z_personal)}")
    print(f"deployment_protected_anomaly: {len(z_safe)}")
    print()
    print(f"{'detector':<12} {'retFPR':>8} {'persFPR':>8} {'safeRec':>8} {'margin':>10}")
    for r in results:
        print(f"{str(r['detector']):<12} {float(r['retention_fpr']):8.4f} {float(r['personal_fpr']):8.4f} "
              f"{float(r['safe_recall']):8.4f} {float(r['score_margin']):10.6f}")
    print()
    print("IMPORTANT: detector variants are an audit, not a hyperparameter sweep on A43. Do not select k/nu/gamma by protected-anomaly recall.")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
