from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_METRICS = [
    "personal_fpr",
    "safe_recall",
    "safe_fnr",
    "global_fpr",
    "personal_gain",
    "safety_drop",
    "score_margin",
    "feedback_used",
    "confirmed_cumulative",
]
OPTIONAL_METRICS = [
    "personal_fpr_residual",
    "personal_fpr_fixed",
    "feedback_censoring_gain",
    "personal_gain_fixed",
]


def ci95_normal(x: pd.Series) -> float:
    a = x.dropna().to_numpy(dtype=float)
    if len(a) <= 1:
        return float("nan")
    return float(1.96 * np.std(a, ddof=1) / np.sqrt(len(a)))


def bootstrap_mean_ci(a: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1 or reps <= 0:
        v = float(a[0])
        return v, v
    draws = rng.choice(a, size=(reps, len(a)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(d.mean())
    if len(d) == 1 or reps <= 0:
        return mean, mean, mean
    draws = rng.choice(d, size=(reps, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return mean, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate continual-personalization baseline results across subjects and order seeds")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv")
    ap.add_argument("--subject-out", default="outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_subject_summary.csv")
    ap.add_argument("--paired-out", default="outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_paired_vs_b0.csv")
    ap.add_argument("--baseline-method", default="B0")
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=1337)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    metrics = BASE_METRICS + [m for m in OPTIONAL_METRICS if m in df.columns]
    required = {"subject", "order_seed", "budget", "method", "session", *BASE_METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    if args.baseline_method not in set(df["method"].astype(str)):
        raise ValueError(f"baseline method {args.baseline_method!r} not found in input")

    subject_group = ["subject", "budget", "method", "session"]
    subject_df = df.groupby(subject_group, as_index=False)[metrics].mean(numeric_only=True)

    subject_out = Path(args.subject_out)
    subject_out.parent.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(subject_out, index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    keys = ["budget", "method", "session"]
    chunks = []
    for group_values, g in subject_df.groupby(keys, sort=True):
        row = dict(zip(keys, group_values))
        row["n_subjects"] = int(g["subject"].nunique())
        for m in metrics:
            values = g[m].dropna().to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            row[f"{m}_ci95"] = ci95_normal(g[m])
            lo, hi = bootstrap_mean_ci(values, args.bootstrap_reps, rng)
            row[f"{m}_boot_low"] = lo
            row[f"{m}_boot_high"] = hi
        chunks.append(row)

    summary = pd.DataFrame(chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    final_session = int(subject_df["session"].max())
    final_subject = subject_df[subject_df["session"] == final_session].copy()
    paired_rows = []
    compare_metrics = ["personal_fpr", "safe_recall", "global_fpr", "score_margin"]
    if "personal_fpr_fixed" in metrics:
        compare_metrics.append("personal_fpr_fixed")
    if "personal_gain_fixed" in metrics:
        compare_metrics.append("personal_gain_fixed")

    for budget in sorted(final_subject["budget"].unique()):
        base = final_subject[(final_subject["budget"] == budget) & (final_subject["method"] == args.baseline_method)].set_index("subject")
        if base.empty:
            continue
        for method in sorted(final_subject["method"].unique()):
            if method == args.baseline_method:
                continue
            cur = final_subject[(final_subject["budget"] == budget) & (final_subject["method"] == method)].set_index("subject")
            common = base.index.intersection(cur.index)
            if len(common) == 0:
                continue
            row = {"budget": budget, "method": method, "baseline_method": args.baseline_method, "session": final_session, "n_subjects": len(common)}
            for metric in compare_metrics:
                mean, lo, hi = paired_bootstrap_delta(
                    cur.loc[common, metric].to_numpy(dtype=float),
                    base.loc[common, metric].to_numpy(dtype=float),
                    args.bootstrap_reps,
                    rng,
                )
                row[f"delta_{metric}_vs_baseline"] = mean
                row[f"delta_{metric}_vs_baseline_boot_low"] = lo
                row[f"delta_{metric}_vs_baseline_boot_high"] = hi
            paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)
    paired_out = Path(args.paired_out)
    paired_out.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_out, index=False)

    final = summary[summary["session"] == final_session].copy()
    cols = [
        "budget", "method", "n_subjects",
        "personal_fpr_mean", "safe_recall_mean", "global_fpr_mean",
        "personal_gain_mean", "safety_drop_mean", "score_margin_mean",
    ]
    for c in ["personal_fpr_fixed_mean", "personal_gain_fixed_mean", "feedback_censoring_gain_mean"]:
        if c in final.columns:
            cols.append(c)

    print(f"rows_input: {len(df)}")
    print(f"subjects: {subject_df['subject'].nunique()}")
    print(f"final_session: {final_session}")
    print(f"baseline_method: {args.baseline_method}")
    print(final[cols].sort_values(["budget", "method"]).to_string(index=False))
    print(f"subject_summary: {subject_out}")
    print(f"summary: {out}")
    print(f"paired_vs_baseline: {paired_out}")
    print(f"bootstrap_reps: {args.bootstrap_reps}")


if __name__ == "__main__":
    main()
