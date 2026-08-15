from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.datasets.ntu120 import read_skeleton_file
from src.features.kinematics import FEATURE_NAMES, extract_kinematic_features


def resolve_path(root: str | Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(root) / path


def load_manifest(manifest: str | Path):
    with Path(manifest).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mask(rows, *, split=None, role=None):
    out = np.ones(len(rows), dtype=bool)
    if split is not None:
        out &= np.asarray([r["inner_split"] == split for r in rows])
    if role is not None:
        out &= np.asarray([r["role"] == role for r in rows])
    return out


def q95(x):
    return float(np.quantile(x, 0.95))


def evaluate(name, score_train, score_calib, score_ret, score_personal, score_safe):
    threshold = q95(score_calib)
    ret_fpr = float(np.mean(score_ret > threshold))
    pers_fpr = float(np.mean(score_personal > threshold))
    safe_rec = float(np.mean(score_safe > threshold))
    margin = float(np.mean(score_safe) - np.mean(score_personal))
    return {
        "detector": name,
        "threshold": threshold,
        "retFPR": ret_fpr,
        "persFPR": pers_fpr,
        "safeRec": safe_rec,
        "margin": margin,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether raw NTU skeleton kinematics contain a one-class A43 safety signal")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/session0/kinematic_audit.csv")
    ap.add_argument("--features-out", default="outputs/ntu120_pilot_v0.1/session0/kinematic_features.csv")
    ap.add_argument("--max-train", type=int, default=5000)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    relevant = [
        r for r in rows
        if r["role"] in {"global_normal", "candidate_personal_normal", "protected_anomaly"}
    ]

    feats = []
    kept = []
    for i, r in enumerate(relevant, start=1):
        raw = read_skeleton_file(resolve_path(args.root, r["path"]))
        feats.append(extract_kinematic_features(raw))
        kept.append(r)
        if i % 1000 == 0:
            print(f"processed: {i}/{len(relevant)}")

    X = np.stack(feats).astype(np.float32)
    rows = kept

    m_train = mask(rows, split="encoder_train", role="global_normal")
    m_calib = mask(rows, split="detector_calib", role="global_normal")
    m_ret = mask(rows, split="retention_val", role="global_normal")
    m_personal = mask(rows, split="deployment_test", role="candidate_personal_normal")
    m_safe = mask(rows, split="deployment_test", role="protected_anomaly")

    for name, m in [
        ("train_global_normal", m_train),
        ("calib_global_normal", m_calib),
        ("retention_global_normal", m_ret),
        ("deployment_personal_normal", m_personal),
        ("deployment_protected_anomaly", m_safe),
    ]:
        print(f"{name}: {int(m.sum())}")
        if m.sum() == 0:
            raise RuntimeError(f"empty partition: {name}")

    scaler = StandardScaler().fit(X[m_train])
    Z = scaler.transform(X).astype(np.float32)
    z_train = Z[m_train]
    z_calib = Z[m_calib]
    z_ret = Z[m_ret]
    z_personal = Z[m_personal]
    z_safe = Z[m_safe]

    if len(z_train) > args.max_train:
        rng = np.random.default_rng(1337)
        z_fit = z_train[rng.choice(len(z_train), args.max_train, replace=False)]
    else:
        z_fit = z_train

    results = []

    center = z_fit.mean(axis=0)
    dist = lambda z: np.linalg.norm(z - center[None, :], axis=1)
    results.append(evaluate("kin_centroid", dist(z_fit), dist(z_calib), dist(z_ret), dist(z_personal), dist(z_safe)))

    cov = LedoitWolf().fit(z_fit)
    maha = lambda z: cov.mahalanobis(z)
    results.append(evaluate("kin_mahalanobis", maha(z_fit), maha(z_calib), maha(z_ret), maha(z_personal), maha(z_safe)))

    nn = NearestNeighbors(n_neighbors=20).fit(z_fit)
    knn = lambda z: nn.kneighbors(z, return_distance=True)[0].mean(axis=1)
    results.append(evaluate("kin_knn20", knn(z_fit), knn(z_calib), knn(z_ret), knn(z_personal), knn(z_safe)))

    svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(z_fit)
    svm_score = lambda z: -svm.decision_function(z).reshape(-1)
    results.append(evaluate("kin_ocsvm", svm_score(z_fit), svm_score(z_calib), svm_score(z_ret), svm_score(z_personal), svm_score(z_safe)))

    print()
    print(f"{'detector':18s} {'retFPR':>8s} {'persFPR':>8s} {'safeRec':>8s} {'margin':>10s}")
    for r in results:
        print(f"{r['detector']:18s} {r['retFPR']:8.4f} {r['persFPR']:8.4f} {r['safeRec']:8.4f} {r['margin']:10.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    features_out = Path(args.features_out)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "action", "setup", "camera", "repetition", "inner_split", "role", "path"] + FEATURE_NAMES
    with features_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, feat in zip(rows, X):
            item = {k: r[k] for k in fields if k in r}
            item.update({name: float(v) for name, v in zip(FEATURE_NAMES, feat)})
            w.writerow(item)

    print()
    print("IMPORTANT: this is a diagnostic audit. A43 is evaluation-only and must not be used to choose detector hyperparameters.")
    print(f"output: {out}")
    print(f"features: {features_out}")


if __name__ == "__main__":
    main()
