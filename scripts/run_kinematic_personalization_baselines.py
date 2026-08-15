from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.features.kinematics import FEATURE_NAMES


def load_table(path: str | Path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.asarray([[float(r[name]) for name in FEATURE_NAMES] for r in rows], dtype=np.float32)
    return rows, x


def mask_rows(rows: List[Dict[str, str]], **conditions) -> np.ndarray:
    m = np.ones(len(rows), dtype=bool)
    for k, v in conditions.items():
        m &= np.asarray([r[k] == str(v) for r in rows], dtype=bool)
    return m


def split_sessions(indices: np.ndarray, sessions: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)
    return [a.astype(np.int64, copy=False) for a in np.array_split(idx, sessions)]


def fit_ocsvm(z: np.ndarray, nu: float, gamma: str):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(z)
    return model


def score(model, z: np.ndarray) -> np.ndarray:
    return -model.decision_function(z).reshape(-1)


def qthreshold(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def evaluate(p_scores, s_scores, g_scores, threshold):
    p_fpr = float(np.mean(p_scores > threshold)) if len(p_scores) else float("nan")
    s_rec = float(np.mean(s_scores > threshold)) if len(s_scores) else float("nan")
    g_fpr = float(np.mean(g_scores > threshold)) if len(g_scores) else float("nan")
    p_mean = float(np.mean(p_scores)) if len(p_scores) else float("nan")
    s_mean = float(np.mean(s_scores)) if len(s_scores) else float("nan")
    return {
        "personal_fpr": p_fpr,
        "safe_recall": s_rec,
        "safe_fnr": 1.0 - s_rec if np.isfinite(s_rec) else float("nan"),
        "global_fpr": g_fpr,
        "personal_score_mean": p_mean,
        "safe_score_mean": s_mean,
        "score_margin": s_mean - p_mean,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled continual-personalization audit on fixed kinematic features")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/baselines/kinematic_personalization.csv")
    ap.add_argument("--methods", nargs="+", default=["K0", "K1", "K2"], choices=["K0", "K1", "K2"])
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--order-seeds", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--ocsvm-nu", type=float, default=0.05)
    ap.add_argument("--ocsvm-gamma", default="scale")
    ap.add_argument("--max-global-train", type=int, default=5000)
    args = ap.parse_args()

    rows, x = load_table(args.features)
    m_train = mask_rows(rows, inner_split="encoder_train", role="global_normal")
    m_calib = mask_rows(rows, inner_split="detector_calib", role="global_normal")
    m_ret = mask_rows(rows, inner_split="retention_val", role="global_normal")
    if min(int(m_train.sum()), int(m_calib.sum()), int(m_ret.sum())) == 0:
        raise RuntimeError("empty train/calibration/retention partition")

    scaler = StandardScaler().fit(x[m_train])
    z = scaler.transform(x).astype(np.float32)
    z_train = z[m_train]
    z_calib = z[m_calib]
    z_ret = z[m_ret]

    if len(z_train) > args.max_global_train:
        rng = np.random.default_rng(1337)
        keep = rng.choice(len(z_train), args.max_global_train, replace=False)
        z_fit = z_train[keep]
    else:
        z_fit = z_train

    base_model = fit_ocsvm(z_fit, args.ocsvm_nu, args.ocsvm_gamma)
    base_threshold = qthreshold(score(base_model, z_calib), args.threshold_quantile)
    base_ret_scores = score(base_model, z_ret)

    deploy_subjects = sorted({int(r["subject"]) for r in rows if r["inner_split"] == "deployment_test"})
    output = []

    for subject in deploy_subjects:
        p_idx = np.flatnonzero(mask_rows(rows, inner_split="deployment_test", role="candidate_personal_normal", subject=subject))
        s_idx = np.flatnonzero(mask_rows(rows, inner_split="deployment_test", role="protected_anomaly", subject=subject))
        if len(p_idx) == 0 or len(s_idx) == 0:
            continue

        base_p_scores_all = score(base_model, z[p_idx])
        base_s_scores = score(base_model, z[s_idx])
        base_met = evaluate(base_p_scores_all, base_s_scores, base_ret_scores, base_threshold)
        base_fixed_fpr = base_met["personal_fpr"]

        for seed in args.order_seeds:
            sessions = split_sessions(p_idx, args.sessions, seed + subject * 1009)
            for budget in args.budgets:
                # Shared caregiver-feedback trajectory: alerts are defined by frozen K0.
                chosen_by_session = []
                for t, arrival in enumerate(sessions, start=1):
                    if len(arrival):
                        alert_mask = score(base_model, z[arrival]) > base_threshold
                        candidates = arrival[alert_mask]
                    else:
                        candidates = np.empty(0, dtype=np.int64)
                    rng = np.random.default_rng(seed * 1000003 + subject * 9176 + budget * 101 + t)
                    chosen = rng.choice(candidates, size=budget, replace=False) if len(candidates) > budget else candidates
                    chosen_by_session.append((candidates, np.asarray(chosen, dtype=np.int64)))

                for method in args.methods:
                    threshold = base_threshold
                    model = base_model
                    confirmed: List[int] = []

                    output.append({
                        "subject": subject, "order_seed": seed, "budget": budget, "method": method, "session": 0,
                        "feedback_available": 0, "feedback_used": 0, "confirmed_cumulative": 0,
                        "remaining_personal": len(p_idx), "threshold": threshold,
                        **base_met,
                        "personal_fpr_residual": base_fixed_fpr,
                        "personal_fpr_fixed": base_fixed_fpr,
                        "feedback_censoring_gain": 0.0,
                        "personal_gain": 0.0,
                        "personal_gain_fixed": 0.0,
                        "safety_drop": 0.0,
                    })

                    for t, (candidates, chosen) in enumerate(chosen_by_session, start=1):
                        confirmed.extend(int(i) for i in chosen)

                        if method == "K1" and confirmed:
                            c_scores = score(base_model, z[np.asarray(confirmed, dtype=np.int64)])
                            threshold = max(base_threshold, qthreshold(c_scores, args.threshold_quantile))
                            model = base_model
                        elif method == "K2" and confirmed:
                            fit_z = np.concatenate([z_fit, z[np.asarray(confirmed, dtype=np.int64)]], axis=0)
                            model = fit_ocsvm(fit_z, args.ocsvm_nu, args.ocsvm_gamma)
                            threshold = qthreshold(score(model, z_calib), args.threshold_quantile)

                        confirmed_set = set(confirmed)
                        remaining = np.asarray([i for i in p_idx if int(i) not in confirmed_set], dtype=np.int64)
                        p_eval = remaining if len(remaining) else p_idx

                        met = evaluate(score(model, z[p_eval]), score(model, z[s_idx]), score(model, z_ret), threshold)
                        fixed_fpr = float(np.mean(score(model, z[p_idx]) > threshold))
                        residual_fpr = met["personal_fpr"]
                        output.append({
                            "subject": subject, "order_seed": seed, "budget": budget, "method": method, "session": t,
                            "feedback_available": len(candidates), "feedback_used": len(chosen),
                            "confirmed_cumulative": len(confirmed), "remaining_personal": len(remaining),
                            "threshold": threshold, **met,
                            "personal_fpr_residual": residual_fpr,
                            "personal_fpr_fixed": fixed_fpr,
                            "feedback_censoring_gain": fixed_fpr - residual_fpr,
                            "personal_gain": base_fixed_fpr - residual_fpr,
                            "personal_gain_fixed": base_fixed_fpr - fixed_fpr,
                            "safety_drop": base_met["safe_recall"] - met["safe_recall"],
                        })

    if not output:
        raise RuntimeError("no results produced")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        w.writeheader(); w.writerows(output)

    personal_counts = [
        sum(1 for r in rows if r["inner_split"] == "deployment_test" and r["role"] == "candidate_personal_normal" and int(r["subject"]) == s)
        for s in deploy_subjects
    ]
    print(f"subjects: {len(deploy_subjects)}")
    print(f"personal_samples_per_subject: min={min(personal_counts)} median={float(np.median(personal_counts)):.1f} max={max(personal_counts)}")
    print("feedback_regime: shared frozen-K0 alerts (controlled comparison)")
    print("K0=no adaptation; K1=threshold-only; K2=OCSVM refit with confirmed personal normals")
    print("metric_note: personal_fpr is the residual unconfirmed pool; personal_fpr_fixed evaluates the same full A42 pool at every session")
    print(f"rows_written: {len(output)}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
