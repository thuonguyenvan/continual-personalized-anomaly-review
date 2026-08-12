from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.metrics.anomaly_metrics import binary_metrics, score_margin, threshold_from_quantile
from src.models.prototype_detector import PrototypeDistanceDetector
from src.models.simple_skeleton_encoder import SimpleSkeletonEncoder


def collate(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    meta = [{k: v for k, v in item.items() if k != "x"} for item in batch]
    return x, meta


@torch.no_grad()
def embed_loader(model, loader, device):
    embeddings = []
    metadata = []
    for x, meta in loader:
        z = model(x.to(device)).cpu()
        embeddings.append(z)
        metadata.extend(meta)
    if not embeddings:
        return torch.empty(0, 128), []
    return torch.cat(embeddings, dim=0), metadata


def main():
    parser = argparse.ArgumentParser(description="Run Session-0 NTU120 gap-validation audit")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/session0_scores.csv")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--checkpoint", required=True, help="Trained encoder checkpoint; random encoders are not allowed for the scientific pilot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SimpleSkeletonEncoder().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()

    global_train = NTUManifestDataset(
        args.manifest,
        args.root,
        inner_split="dev_train",
        roles=["global_normal"],
        seq_len=args.seq_len,
    )
    global_val = NTUManifestDataset(
        args.manifest,
        args.root,
        inner_split="dev_val",
        roles=["global_normal"],
        seq_len=args.seq_len,
    )
    deployment = NTUManifestDataset(
        args.manifest,
        args.root,
        inner_split="deployment_test",
        roles=["global_normal", "candidate_personal_normal", "protected_anomaly"],
        seq_len=args.seq_len,
    )

    dl_args = dict(batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    z_train, _ = embed_loader(encoder, DataLoader(global_train, **dl_args), device)
    z_val, _ = embed_loader(encoder, DataLoader(global_val, **dl_args), device)
    z_test, meta_test = embed_loader(encoder, DataLoader(deployment, **dl_args), device)

    if z_train.numel() == 0 or z_val.numel() == 0 or z_test.numel() == 0:
        raise RuntimeError("One or more required dataset partitions are empty. Check manifest roles/splits.")

    detector = PrototypeDistanceDetector()
    detector.fit(z_train)

    val_scores = detector.score(z_val).cpu().numpy()
    threshold = threshold_from_quantile(val_scores, q=args.threshold_quantile)
    test_scores = detector.score(z_test).cpu().numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for score, meta in zip(test_scores, meta_test):
        row = dict(meta)
        row["score"] = float(score)
        row["predicted_anomaly"] = int(score > threshold)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    global_scores = [r["score"] for r in rows if r["role"] == "global_normal"]
    personal_scores = [r["score"] for r in rows if r["role"] == "candidate_personal_normal"]
    protected_scores = [r["score"] for r in rows if r["role"] == "protected_anomaly"]

    if personal_scores and protected_scores:
        metrics = binary_metrics(personal_scores, protected_scores, threshold)
        margins = score_margin(personal_scores, protected_scores)
        print("Session 0 candidate-personal-normal vs protected-anomaly metrics")
        for k, v in {**metrics, **margins}.items():
            print(f"{k}: {v}")

    if global_scores:
        global_fpr = sum(s > threshold for s in global_scores) / len(global_scores)
        print(f"global_normal_fpr: {global_fpr:.6f}")

    print(f"threshold: {threshold:.6f}")
    print(f"saved_scores: {out_path}")


if __name__ == "__main__":
    main()
