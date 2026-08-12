from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.metrics.anomaly_metrics import binary_metrics, score_margin, threshold_from_quantile
from src.models.prototype_detector import PrototypeDetector
from src.models.simple_skeleton_encoder import SimpleSkeletonEncoder
from src.utils.checkpoint import load_encoder_checkpoint


def collate(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    meta = [{k: v for k, v in item.items() if k != "x"} for item in batch]
    return x, meta


@torch.no_grad()
def embed_loader(model, loader, device):
    embeddings, metadata = [], []
    for x, meta in loader:
        embeddings.append(model(x.to(device)).cpu()); metadata.extend(meta)
    if not embeddings: return torch.empty(0, 128), []
    return torch.cat(embeddings, dim=0), metadata


def make_loader(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)


def main():
    parser = argparse.ArgumentParser(description="Run leakage-resistant Session-0 NTU120 gap-validation audit")
    parser.add_argument("--manifest", required=True); parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/ntu120_pilot_v0.1/session0/session0_scores.csv")
    parser.add_argument("--seq-len", type=int, default=64); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold-quantile", type=float, default=0.95); parser.add_argument("--checkpoint", required=True, help="Trained encoder checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SimpleSkeletonEncoder().to(device); state, checkpoint_metadata = load_encoder_checkpoint(args.checkpoint, map_location=device); encoder.load_state_dict(state); encoder.eval()
    if checkpoint_metadata: print(f"checkpoint_metadata: {checkpoint_metadata}")

    global_train = NTUManifestDataset(args.manifest, args.root, inner_split="encoder_train", roles=["global_normal"], seq_len=args.seq_len)
    detector_calib = NTUManifestDataset(args.manifest, args.root, inner_split="detector_calib", roles=["global_normal"], seq_len=args.seq_len)
    retention = NTUManifestDataset(args.manifest, args.root, inner_split="retention_val", roles=["global_normal"], seq_len=args.seq_len)
    deployment = NTUManifestDataset(args.manifest, args.root, inner_split="deployment_test", roles=["global_normal", "candidate_personal_normal", "protected_anomaly"], seq_len=args.seq_len)

    z_train, _ = embed_loader(encoder, make_loader(global_train, args.batch_size), device)
    z_calib, _ = embed_loader(encoder, make_loader(detector_calib, args.batch_size), device)
    z_retention, _ = embed_loader(encoder, make_loader(retention, args.batch_size), device)
    z_test, meta_test = embed_loader(encoder, make_loader(deployment, args.batch_size), device)
    if any(x.numel() == 0 for x in [z_train, z_calib, z_retention, z_test]): raise RuntimeError("One or more required partitions are empty. Regenerate the manifest.")

    detector = PrototypeDetector(); detector.fit(z_train.numpy(), threshold_quantile=args.threshold_quantile)
    threshold = threshold_from_quantile(detector.score_samples(z_calib.numpy()), q=args.threshold_quantile)
    retention_scores = detector.score_samples(z_retention.numpy()); test_scores = detector.score_samples(z_test.numpy())

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True); rows: List[Dict[str, object]] = []
    for score, meta in zip(test_scores, meta_test):
        row = dict(meta); row["score"] = float(score); row["predicted_anomaly"] = int(score > threshold); rows.append(row)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    deploy_global_scores = [r["score"] for r in rows if r["role"] == "global_normal"]
    personal_scores = [r["score"] for r in rows if r["role"] == "candidate_personal_normal"]
    protected_scores = [r["score"] for r in rows if r["role"] == "protected_anomaly"]
    if personal_scores and protected_scores:
        metrics = binary_metrics(personal_scores, protected_scores, threshold); margins = score_margin(personal_scores, protected_scores)
        print("Session 0 candidate-personal-normal vs protected-anomaly metrics")
        for k, v in {**metrics, **margins}.items(): print(f"{k}: {v}")
    print(f"retention_global_fpr: {float((retention_scores > threshold).mean()):.6f}")
    if deploy_global_scores: print(f"deployment_global_fpr: {sum(s > threshold for s in deploy_global_scores) / len(deploy_global_scores):.6f}")
    print(f"threshold: {threshold:.6f}"); print(f"saved_scores: {out_path}")


if __name__ == "__main__": main()
