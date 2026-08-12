from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.models.simple_skeleton_encoder import SimpleSkeletonEncoder
from src.utils.checkpoint import file_sha256, git_commit, load_encoder_checkpoint, write_json


def collate(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    meta = [{k: v for k, v in item.items() if k != "x"} for item in batch]
    return x, meta


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frozen NTU120 skeleton embeddings once for CPU-side baseline experiments")
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="outputs/ntu120_embeddings.npz"); ap.add_argument("--metadata-out", default="outputs/ntu120_embeddings_metadata.csv")
    ap.add_argument("--provenance-out", default=None); ap.add_argument("--seq-len", type=int, default=64); ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device: {device}")
    model = SimpleSkeletonEncoder().to(device); state, checkpoint_metadata = load_encoder_checkpoint(args.checkpoint, map_location=device); model.load_state_dict(state); model.eval()

    ds = NTUManifestDataset(args.manifest, args.root, roles=["global_normal", "candidate_personal_normal", "protected_anomaly"], seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    embeddings, metadata = [], []
    for i, (x, meta) in enumerate(loader, start=1):
        embeddings.append(model(x.to(device)).cpu().numpy().astype(np.float32, copy=False)); metadata.extend(meta)
        if i % 100 == 0: print(f"batches: {i}/{len(loader)}")
    if not embeddings: raise RuntimeError("No experiment-relevant samples found")

    z = np.concatenate(embeddings, axis=0); out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, embeddings=z)
    metadata_out = Path(args.metadata_out); metadata_out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "action", "setup", "camera", "repetition", "outer_split", "inner_split", "role", "path"]
    with metadata_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(metadata)
    if len(metadata) != z.shape[0]: raise RuntimeError(f"metadata/embedding length mismatch: {len(metadata)} vs {z.shape[0]}")

    provenance_out = Path(args.provenance_out) if args.provenance_out else out.with_suffix(".provenance.json")
    write_json(provenance_out, {"git_commit": git_commit(), "manifest_sha256": file_sha256(args.manifest), "checkpoint_sha256": file_sha256(args.checkpoint), "checkpoint_metadata": checkpoint_metadata, "seq_len": args.seq_len, "samples": int(z.shape[0]), "embedding_dim": int(z.shape[1]), "embeddings_sha256": file_sha256(out), "metadata_sha256": file_sha256(metadata_out), "embeddings": str(out), "metadata": str(metadata_out)})
    print(f"samples: {z.shape[0]}"); print(f"embedding_dim: {z.shape[1]}"); print(f"embeddings: {out}"); print(f"metadata: {metadata_out}"); print(f"provenance: {provenance_out}")


if __name__ == "__main__": main()
