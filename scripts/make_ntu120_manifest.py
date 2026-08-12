"""Create a CSV manifest for the NTU RGB+D 120 skeleton pilot.

The official NTU120 cross-subject training subjects are further partitioned by
subject into four disjoint sets so model selection, anomaly-threshold
calibration, and global-normal retention evaluation do not reuse the same
subjects.
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


def make_inner_splits(
    train_subjects: list[int],
    seed: int,
    n_encoder_val: int,
    n_detector_calib: int,
    n_retention_val: int,
) -> tuple[set[int], set[int], set[int], set[int]]:
    ids = list(train_subjects)
    required_holdout = n_encoder_val + n_detector_calib + n_retention_val
    if min(n_encoder_val, n_detector_calib, n_retention_val) < 1:
        raise ValueError("all holdout counts must be >= 1")
    if required_holdout >= len(ids):
        raise ValueError("holdout subject counts leave no encoder-training subjects")

    rng = random.Random(seed)
    rng.shuffle(ids)

    p = 0
    encoder_val = set(ids[p : p + n_encoder_val])
    p += n_encoder_val
    detector_calib = set(ids[p : p + n_detector_calib])
    p += n_detector_calib
    retention_val = set(ids[p : p + n_retention_val])
    p += n_retention_val
    encoder_train = set(ids[p:])
    return encoder_train, encoder_val, detector_calib, retention_val


def main() -> None:
    ap = argparse.ArgumentParser(description="Create leakage-resistant NTU120 pilot manifest")
    ap.add_argument("--skeleton-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-encoder-val", type=int, default=6)
    ap.add_argument("--n-detector-calib", type=int, default=5)
    ap.add_argument("--n-retention-val", type=int, default=5)
    args = ap.parse_args()

    metadata = scan_skeleton_files(args.skeleton_root)
    encoder_train, encoder_val, detector_calib, retention_val = make_inner_splits(
        sorted(CSUB_TRAIN),
        args.seed,
        args.n_encoder_val,
        args.n_detector_calib,
        args.n_retention_val,
    )

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
            elif m.subject in encoder_val:
                inner = "encoder_val"
            elif m.subject in detector_calib:
                inner = "detector_calib"
            elif m.subject in retention_val:
                inner = "retention_val"
            else:
                inner = "encoder_train"
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
    print(f"encoder_train subjects ({len(encoder_train)}): {sorted(encoder_train)}")
    print(f"encoder_val subjects ({len(encoder_val)}): {sorted(encoder_val)}")
    print(f"detector_calib subjects ({len(detector_calib)}): {sorted(detector_calib)}")
    print(f"retention_val subjects ({len(retention_val)}): {sorted(retention_val)}")


if __name__ == "__main__":
    main()
