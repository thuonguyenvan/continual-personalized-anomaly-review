from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_line(summary: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for (method, budget), g in summary.groupby(["method", "budget"], sort=True):
        g = g.sort_values("session")
        x = g["session"]
        y = g[f"{metric}_mean"]
        e = g.get(f"{metric}_ci95")
        label = f"{method}, K={budget}"
        if e is not None:
            plt.errorbar(x, y, yerr=e, marker="o", capsize=2, label=label)
        else:
            plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Personalization session")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()


def save_tradeoff(summary: pd.DataFrame, out: Path) -> None:
    final = summary[summary["session"] == summary["session"].max()].copy()
    plt.figure(figsize=(6, 5))
    for (method, budget), g in final.groupby(["method", "budget"], sort=True):
        x = float(g["personal_gain_mean"].iloc[0])
        y = float(g["safety_drop_mean"].iloc[0])
        plt.scatter([x], [y])
        plt.annotate(f"{method}, K={budget}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("PersonalGain = FPR_personal(M0) - FPR_personal(Mt)")
    plt.ylabel("SafetyDrop = Recall_safe(M0) - Recall_safe(Mt)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate core NTU120 personalization pilot figures")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", default="outputs/ntu120_pilot_v0.1/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    out = Path(args.out_dir)
    save_line(df, "personal_fpr", "Personal-normal FPR", out / "personal_fpr_by_session.png")
    save_line(df, "safe_recall", "Protected-anomaly recall", out / "safe_recall_by_session.png")
    save_line(df, "global_fpr", "Retained global-normal FPR", out / "global_fpr_by_session.png")
    save_tradeoff(df, out / "personal_gain_vs_safety_drop.png")
    print(f"figures: {out}")


if __name__ == "__main__":
    main()
