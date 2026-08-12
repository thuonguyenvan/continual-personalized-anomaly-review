from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_ROWS = 114480
EXPECTED_COLUMNS = {
    "path", "setup", "camera", "subject", "repetition", "action",
    "outer_split", "inner_split", "role",
}
VALID_INNER = {
    "encoder_train", "encoder_val", "detector_calib", "retention_val", "deployment_test"
}
VALID_OUTER = {"train", "test"}
VALID_ROLES = {
    "global_normal", "candidate_personal_normal", "protected_anomaly", "excluded"
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate NTU120 manifest before expensive runs")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None, help="Optional dataset root for path existence checks")
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    root = Path(args.root) if args.root else None

    with manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = EXPECTED_COLUMNS - cols
        if missing:
            raise SystemExit(f"FAIL: missing columns: {sorted(missing)}")
        rows = list(reader)

    errors: list[str] = []

    if len(rows) != args.expected_rows:
        errors.append(f"row count {len(rows)} != expected {args.expected_rows}")

    paths = [r["path"] for r in rows]
    duplicate_paths = len(paths) - len(set(paths))
    if duplicate_paths:
        errors.append(f"duplicate manifest paths: {duplicate_paths}")

    outer_counts = Counter(r["outer_split"] for r in rows)
    inner_counts = Counter(r["inner_split"] for r in rows)
    role_counts = Counter(r["role"] for r in rows)

    bad_outer = sorted(set(outer_counts) - VALID_OUTER)
    bad_inner = sorted(set(inner_counts) - VALID_INNER)
    bad_roles = sorted(set(role_counts) - VALID_ROLES)
    if bad_outer:
        errors.append(f"invalid outer_split values: {bad_outer}")
    if bad_inner:
        errors.append(f"invalid inner_split values: {bad_inner}")
    if bad_roles:
        errors.append(f"invalid role values: {bad_roles}")

    missing_inner = sorted(VALID_INNER - set(inner_counts))
    if missing_inner:
        errors.append(f"missing required inner splits: {missing_inner}")

    subject_inner: dict[int, set[str]] = defaultdict(set)
    subject_outer: dict[int, set[str]] = defaultdict(set)
    action_roles: dict[int, set[str]] = defaultdict(set)

    for r in rows:
        s = int(r["subject"])
        a = int(r["action"])
        subject_inner[s].add(r["inner_split"])
        subject_outer[s].add(r["outer_split"])
        action_roles[a].add(r["role"])

        if r["outer_split"] == "test" and r["inner_split"] != "deployment_test":
            errors.append(f"test row is not deployment_test: {r['path']}")
            if len(errors) > 20:
                break
        if r["outer_split"] == "train" and r["inner_split"] == "deployment_test":
            errors.append(f"train row marked deployment_test: {r['path']}")
            if len(errors) > 20:
                break

    # Every subject must belong to exactly one inner split. This is stricter than
    # merely checking train/test leakage and protects calibration/evaluation roles.
    mixed_inner = {s: x for s, x in subject_inner.items() if len(x) > 1}
    if mixed_inner:
        errors.append(f"subjects assigned to multiple inner splits: {mixed_inner}")

    mixed_outer = {s: x for s, x in subject_outer.items() if len(x) > 1}
    if mixed_outer:
        errors.append(f"subjects assigned to multiple outer splits: {mixed_outer}")

    for action, expected_role in [(42, "candidate_personal_normal"), (43, "protected_anomaly")]:
        actual = action_roles.get(action, set())
        if actual != {expected_role}:
            errors.append(f"A{action:03d} roles {sorted(actual)} != [{expected_role}]")

    if root is not None:
        missing_files = []
        for p in paths:
            pp = Path(p)
            resolved = pp if pp.is_absolute() else root / pp
            if not resolved.exists():
                missing_files.append(str(resolved))
                if len(missing_files) >= 10:
                    break
        if missing_files:
            errors.append("missing skeleton files (first up to 10): " + "; ".join(missing_files))

    subject_split_counts = Counter(next(iter(v)) for v in subject_inner.values())

    print(f"rows: {len(rows)}")
    print(f"subjects: {len(subject_inner)}")
    print(f"actions: {len(action_roles)}")
    print(f"outer_split_counts: {dict(sorted(outer_counts.items()))}")
    print(f"inner_split_counts: {dict(sorted(inner_counts.items()))}")
    print(f"subject_inner_split_counts: {dict(sorted(subject_split_counts.items()))}")
    print(f"role_counts: {dict(sorted(role_counts.items()))}")
    print(f"duplicate_paths: {duplicate_paths}")

    if errors:
        print("\nPRECHECK FAILED")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("\nPRECHECK PASSED")


if __name__ == "__main__":
    main()
