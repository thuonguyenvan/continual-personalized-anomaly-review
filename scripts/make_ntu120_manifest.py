"""Create a CSV manifest for the NTU RGB+D 120 skeleton pilot.

Usage:
    python scripts/make_ntu120_manifest.py \
        --skeleton-root /path/to/nturgbd120_skeletons \
        --output data/ntu120_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from src.datasets.ntu120 import scan_skeleton_files


CSUB_TRAIN = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80,
    81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103,
}

GLOBAL_NORMAL = {1, 2, 6, 8, 9, 11, 12, 23, 25, 28, 29, 33, 34, 35, 36, 37, 40, 80, 96, 97, 98, 99, 100, 101}
PERSONAL_NORMAL = {42}
PROTECTED_ANOMALY = {43}


def role(action: int) -> str:
    if action in GLOBAL_NORMAL:
        return "global_normal"
    if action in PERSONAL_NORMAL:
        return "candidate_personal_normal"
    if action in PROTECTED_ANOMALY:
        return "protected_anomaly"
    return "excluded"


def make_inner_split(train_subjects: list[int], seed: int, val_fraction: float) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    ids = list(train_subjects)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    val = set(sorted(ids[:n_val]))
    train = set(sorted(ids[n_val:]))
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    metadata = scan_skeleton_files(args.skeleton_root)
    inner_train, inner_val = make_inner_split(sorted(CSUB_TRAIN), args.seed, args.val_fraction)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path", "setup", "camera", "subject", "repetition", "action",
        "outer_split", "inner_split", "role",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metadata:
            outer = "train" if m.subject in CSUB_TRAIN else "test"
            if outer == "test":
                inner = "deployment_test"
            elif m.subject in inner_val:
                inner = "dev_val"
            else:
                inner = "dev_train"
            writer.writerow({
                "path": m.path,
                "setup": m.setup,
                "camera": m.camera,
                "subject": m.subject,
                "repetition": m.repetition,
                "action": m.action,
                "outer_split": outer,
                "inner_split": inner,
                "role": role(m.action),
            })

    print(f"Wrote {len(metadata)} rows to {out}")
    print(f"Inner dev-train subjects ({len(inner_train)}): {sorted(inner_train)}")
    print(f"Inner dev-val subjects ({len(inner_val)}): {sorted(inner_val)}")


if __name__ == "__main__":
    main()
