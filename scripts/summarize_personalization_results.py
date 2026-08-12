from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
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


def ci95(x: pd.Series) -> float:
    a = x.dropna().to_numpy(dtype=float)
    if len(a) <= 1:
        return float("nan")
    return float(1.96 * np.std(a, ddof=1) / np.sqrt(len(a)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate continual-personalization baseline results across subjects and order seeds")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv")
    ap.add_argument("--subject-out", default="outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_subject_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"subject", "order_seed", "budget", "method", "session", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")

    # First average repeated orderings within each subject. This prevents subjects
    # with more stochastic repeats from receiving larger statistical weight.
    subject_group = ["subject", "budget", "method", "session"]
    subject_df = df.groupby(subject_group, as_index=False)[METRICS].mean(numeric_only=True)

    subject_out = Path(args.subject_out)
    subject_out.parent.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(subject_out, index=False)

    keys = ["budget", "method", "session"]
    chunks = []
    for group_values, g in subject_df.groupby(keys, sort=True):
        row = dict(zip(keys, group_values))
        row["n_subjects"] = int(g["subject"].nunique())
        for m in METRICS:
            values = g[m]
            row[f"{m}_mean"] = float(values.mean())
            row[f"{m}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{m}_ci95"] = ci95(values)
        chunks.append(row)

    summary = pd.DataFrame(chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    final_session = int(summary["session"].max())
    final = summary[summary["session"] == final_session].copy()
    cols = [
        "budget", "method", "n_subjects",
        "personal_fpr_mean", "safe_recall_mean", "global_fpr_mean",
        "personal_gain_mean", "safety_drop_mean", "score_margin_mean",
    ]
    print(f"rows_input: {len(df)}")
    print(f"subjects: {subject_df['subject'].nunique()}")
    print(f"final_session: {final_session}")
    print(final[cols].sort_values(["budget", "method"]).to_string(index=False))
    print(f"subject_summary: {subject_out}")
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
