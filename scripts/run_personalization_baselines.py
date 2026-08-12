from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from sklearn.svm import OneClassSVM
except ImportError:  # optional dependency until B3 is requested
    OneClassSVM = None


@dataclass
class EmbeddingTable:
    z: np.ndarray
    rows: List[Dict[str, str]]


def load_embeddings(npz_path: str | Path, metadata_path: str | Path) -> EmbeddingTable:
    z = np.load(npz_path)["embeddings"].astype(np.float32, copy=False)
    with Path(metadata_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(z):
        raise ValueError(f"metadata/embedding mismatch: {len(rows)} vs {len(z)}")
    return EmbeddingTable(z=z, rows=rows)


def mask_rows(rows: List[Dict[str, str]], **conditions) -> np.ndarray:
    mask = np.ones(len(rows), dtype=bool)
    for key, value in conditions.items():
        if isinstance(value, (set, list, tuple)):
            allowed = {str(v) for v in value}
            mask &= np.asarray([r[key] in allowed for r in rows], dtype=bool)
        else:
            mask &= np.asarray([r[key] == str(value) for r in rows], dtype=bool)
    return mask


def centroid(z: np.ndarray) -> np.ndarray:
    if len(z) == 0:
        raise ValueError("cannot compute centroid of empty array")
    return z.mean(axis=0)


def distance_scores(z: np.ndarray, prototypes: Iterable[np.ndarray]) -> np.ndarray:
    ps = list(prototypes)
    if not ps:
        raise ValueError("at least one prototype is required")
    d = [np.linalg.norm(z - p[None, :], axis=1) for p in ps]
    return np.min(np.stack(d, axis=1), axis=1)


def quantile_threshold(scores: np.ndarray, q: float) -> float:
    if len(scores) == 0:
        raise ValueError("cannot calibrate threshold on empty scores")
    return float(np.quantile(scores, q))


def split_sessions(indices: np.ndarray, sessions: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)
    return [x.astype(np.int64, copy=False) for x in np.array_split(idx, sessions)]


def safe_rate(scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    if len(scores) == 0:
        return float("nan"), float("nan")
    recall = float(np.mean(scores > threshold))
    return recall, 1.0 - recall


def fpr(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores > threshold)) if len(scores) else float("nan")


def fit_ocsvm(z_train: np.ndarray, nu: float, gamma: str):
    if OneClassSVM is None:
        raise RuntimeError("B3 requires scikit-learn: pip install scikit-learn")
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(z_train)
    return model


def ocsvm_scores(model, z: np.ndarray) -> np.ndarray:
    # sklearn: positive means inlier; convert so larger = more anomalous.
    return -model.decision_function(z).reshape(-1)


def evaluate(scores_personal, scores_safe, scores_global, threshold):
    recall, fnr = safe_rate(scores_safe, threshold)
    p_fpr = fpr(scores_personal, threshold)
    g_fpr = fpr(scores_global, threshold)
    p_mean = float(np.mean(scores_personal)) if len(scores_personal) else float("nan")
    s_mean = float(np.mean(scores_safe)) if len(scores_safe) else float("nan")
    return {
        "personal_fpr": p_fpr,
        "safe_recall": recall,
        "safe_fnr": fnr,
        "global_fpr": g_fpr,
        "personal_score_mean": p_mean,
        "safe_score_mean": s_mean,
        "score_margin": s_mean - p_mean,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run B0-B3 continual-personalization baselines on frozen embeddings")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/baselines/frozen_baselines.csv")
    ap.add_argument("--methods", nargs="+", default=["B0", "B1", "B2", "B3"], choices=["B0", "B1", "B2", "B3"])
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--order-seeds", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--ocsvm-nu", type=float, default=0.05)
    ap.add_argument("--ocsvm-gamma", default="scale")
    ap.add_argument("--ocsvm-max-global-train", type=int, default=5000)
    args = ap.parse_args()

    tab = load_embeddings(args.embeddings, args.metadata)
    rows, z = tab.rows, tab.z

    m_train = mask_rows(rows, inner_split="dev_train", role="global_normal")
    m_val = mask_rows(rows, inner_split="dev_val", role="global_normal")
    m_deploy = mask_rows(rows, inner_split="deployment_test")
    z_train = z[m_train]
    z_val = z[m_val]
    if len(z_train) == 0 or len(z_val) == 0:
        raise RuntimeError("empty global dev_train/dev_val embedding partitions")

    global_proto = centroid(z_train)
    base_val_scores = distance_scores(z_val, [global_proto])
    base_threshold = quantile_threshold(base_val_scores, args.threshold_quantile)

    deploy_subjects = sorted({int(r["subject"]) for i, r in enumerate(rows) if m_deploy[i]})
    output_rows: List[Dict[str, object]] = []

    # Fixed subset for B3 runtime/reproducibility.
    if len(z_train) > args.ocsvm_max_global_train:
        rng_global = np.random.default_rng(1337)
        keep = rng_global.choice(len(z_train), size=args.ocsvm_max_global_train, replace=False)
        z_train_oc = z_train[keep]
    else:
        z_train_oc = z_train

    for subject in deploy_subjects:
        m_personal = mask_rows(rows, inner_split="deployment_test", role="candidate_personal_normal", subject=subject)
        m_safe = mask_rows(rows, inner_split="deployment_test", role="protected_anomaly", subject=subject)
        p_idx = np.flatnonzero(m_personal)
        s_idx = np.flatnonzero(m_safe)
        if len(p_idx) == 0 or len(s_idx) == 0:
            continue

        for seed in args.order_seeds:
            sessions = split_sessions(p_idx, args.sessions, seed + subject * 1009)

            for budget in args.budgets:
                for method in args.methods:
                    confirmed: List[int] = []
                    threshold = base_threshold
                    personal_proto = None
                    oc_model = None

                    def current_scores(indices: np.ndarray) -> np.ndarray:
                        zz = z[indices]
                        if method == "B3":
                            return ocsvm_scores(oc_model, zz)
                        ps = [global_proto] + ([] if personal_proto is None else [personal_proto])
                        return distance_scores(zz, ps)

                    if method == "B3":
                        oc_model = fit_ocsvm(z_train_oc, args.ocsvm_nu, args.ocsvm_gamma)
                        threshold = quantile_threshold(ocsvm_scores(oc_model, z_val), args.threshold_quantile)

                    p_scores = current_scores(p_idx)
                    s_scores = current_scores(s_idx)
                    g_scores = ocsvm_scores(oc_model, z_val) if method == "B3" else distance_scores(z_val, [global_proto] + ([] if personal_proto is None else [personal_proto]))
                    met0 = evaluate(p_scores, s_scores, g_scores, threshold)
                    output_rows.append({
                        "subject": subject, "order_seed": seed, "budget": budget, "method": method,
                        "session": 0, "feedback_available": 0, "feedback_used": 0,
                        "confirmed_cumulative": 0, "remaining_personal": len(p_idx), "threshold": threshold,
                        **met0,
                    })
                    base_personal_fpr = met0["personal_fpr"]
                    base_safe_recall = met0["safe_recall"]

                    for t, arrival in enumerate(sessions, start=1):
                        if len(arrival):
                            arrival_scores = current_scores(arrival)
                            false_alarm_idx = arrival[arrival_scores > threshold]
                        else:
                            false_alarm_idx = np.empty(0, dtype=np.int64)

                        rng = np.random.default_rng(seed * 1000003 + subject * 9176 + budget * 101 + t)
                        if len(false_alarm_idx) > budget:
                            chosen = rng.choice(false_alarm_idx, size=budget, replace=False)
                        else:
                            chosen = false_alarm_idx
                        confirmed.extend(int(x) for x in chosen)

                        if method == "B1" and confirmed:
                            # A threshold-only baseline must actually respond to sparse personal feedback.
                            # Mixing a handful of confirmed samples into thousands of dev-val samples makes
                            # their influence effectively zero. Instead keep the population threshold as a
                            # lower bound and move it to the personal-normal quantile when necessary.
                            c_scores = distance_scores(z[np.asarray(confirmed)], [global_proto])
                            personal_threshold = quantile_threshold(c_scores, args.threshold_quantile)
                            threshold = max(base_threshold, personal_threshold)
                        elif method == "B2" and confirmed:
                            personal_proto = centroid(z[np.asarray(confirmed)])
                            threshold = quantile_threshold(distance_scores(z_val, [global_proto, personal_proto]), args.threshold_quantile)
                        elif method == "B3" and confirmed:
                            fit_z = np.concatenate([z_train_oc, z[np.asarray(confirmed)]], axis=0)
                            oc_model = fit_ocsvm(fit_z, args.ocsvm_nu, args.ocsvm_gamma)
                            threshold = quantile_threshold(ocsvm_scores(oc_model, z_val), args.threshold_quantile)

                        confirmed_set = set(confirmed)
                        remaining = np.asarray([i for i in p_idx if int(i) not in confirmed_set], dtype=np.int64)
                        p_eval = remaining if len(remaining) else p_idx
                        p_scores = current_scores(p_eval)
                        s_scores = current_scores(s_idx)
                        g_scores = ocsvm_scores(oc_model, z_val) if method == "B3" else distance_scores(z_val, [global_proto] + ([] if personal_proto is None else [personal_proto]))
                        met = evaluate(p_scores, s_scores, g_scores, threshold)
                        output_rows.append({
                            "subject": subject, "order_seed": seed, "budget": budget, "method": method,
                            "session": t, "feedback_available": len(false_alarm_idx), "feedback_used": len(chosen),
                            "confirmed_cumulative": len(confirmed), "remaining_personal": len(remaining), "threshold": threshold,
                            **met,
                            "personal_gain": base_personal_fpr - met["personal_fpr"],
                            "safety_drop": base_safe_recall - met["safe_recall"],
                        })

    if not output_rows:
        raise RuntimeError("no baseline results produced")

    for r in output_rows:
        r.setdefault("personal_gain", 0.0)
        r.setdefault("safety_drop", 0.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(output_rows)

    personal_counts = [sum(1 for r in rows if r["inner_split"] == "deployment_test" and r["role"] == "candidate_personal_normal" and int(r["subject"]) == s) for s in deploy_subjects]
    print(f"subjects: {len(deploy_subjects)}")
    print(f"personal_samples_per_subject: min={min(personal_counts)} median={float(np.median(personal_counts)):.1f} max={max(personal_counts)}")
    if max(args.budgets) > max(personal_counts):
        print("WARNING: at least one requested feedback budget exceeds the total personal-normal samples available for a subject; inspect feedback_used and consider smaller NTU-specific budgets.")
    print(f"rows_written: {len(output_rows)}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
