from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.utils.checkpoint import file_sha256, git_commit, load_encoder_checkpoint, write_json

PINNED_INFOGCN_COMMIT = "873feaa85160317335a83e04013e0ffa3f63525e"


def official_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def import_infogcn(repo: Path):
    if not repo.exists():
        raise FileNotFoundError(f"Official InfoGCN repo not found at {repo}. Run scripts/setup_infogcn_official.sh first.")
    sys.path.insert(0, str(repo.resolve()))
    from model.infogcn import InfoGCN
    return InfoGCN


def collate(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    x = x.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()  # [N,C,T,V,M=1]
    meta = [{k: v for k, v in item.items() if k != "x"} for item in batch]
    return x, meta


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract latent z from the official InfoGCN architecture trained under the normal-only protocol")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--official-repo", default="third_party/infogcn_official")
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/embeddings_infogcn/embeddings.npz")
    ap.add_argument("--metadata-out", default="outputs/ntu120_pilot_v0.1/embeddings_infogcn/metadata.csv")
    ap.add_argument("--provenance-out", default="outputs/ntu120_pilot_v0.1/embeddings_infogcn/embeddings.provenance.json")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, meta = load_encoder_checkpoint(args.checkpoint, map_location=device)
    if meta.get("encoder_arch") != "official_infogcn_cvpr2022":
        raise RuntimeError(f"checkpoint is not an official-InfoGCN adapter checkpoint: {meta.get('encoder_arch')!r}")

    repo = Path(args.official_repo)
    InfoGCN = import_infogcn(repo)
    commit = official_commit(repo)
    if meta.get("official_commit") not in (None, commit):
        raise RuntimeError(f"official InfoGCN commit mismatch: checkpoint={meta.get('official_commit')} local={commit}")
    if commit != PINNED_INFOGCN_COMMIT:
        print(f"WARNING: local InfoGCN commit is {commit}, expected pinned {PINNED_INFOGCN_COMMIT}")

    model = InfoGCN(
        num_class=int(meta.get("num_classes", 24)), num_point=25, num_person=int(meta.get("num_person", 1)),
        graph="graph.ntu_rgb_d.Graph", in_channels=3, drop_out=0,
        num_head=int(meta.get("num_head", 3)), noise_ratio=float(meta.get("noise_ratio", 0.5)),
        k=int(meta.get("k", 1)), gain=float(meta.get("z_prior_gain", 3.0)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    seq_len = int(meta.get("seq_len", 64))
    preprocess_mode = str(meta.get("preprocess_mode", "sequence_origin"))
    ds = NTUManifestDataset(
        args.manifest, args.root,
        roles=["global_normal", "candidate_personal_normal", "protected_anomaly"],
        seq_len=seq_len, preprocess_mode=preprocess_mode,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate)

    zs = []
    rows = []
    for i, (x, batch_meta) in enumerate(loader, start=1):
        _, z = model(x.to(device, non_blocking=True))
        zs.append(z.cpu().numpy().astype(np.float32, copy=False))
        rows.extend(batch_meta)
        if i % 50 == 0:
            print(f"batches: {i}/{len(loader)}")

    z = np.concatenate(zs, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=z)

    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "action", "setup", "camera", "repetition", "outer_split", "inner_split", "role", "path"]
    with metadata_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    provenance_out = Path(args.provenance_out)
    write_json(provenance_out, {
        "encoder_arch": "official_infogcn_cvpr2022",
        "official_repo": "stnoah1/infogcn",
        "official_commit": commit,
        "project_git_commit_at_extraction": git_commit(),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_metadata": meta,
        "seq_len": seq_len,
        "preprocess_mode": preprocess_mode,
        "samples": int(z.shape[0]),
        "embedding_dim": int(z.shape[1]),
        "embeddings": str(out),
        "embeddings_sha256": file_sha256(out),
        "metadata": str(metadata_out),
        "metadata_sha256": file_sha256(metadata_out),
    })

    print(f"device: {device}")
    print(f"official_infogcn_commit: {commit}")
    print(f"samples: {z.shape[0]}")
    print(f"embedding_dim: {z.shape[1]}")
    print(f"embeddings: {out}")
    print(f"metadata: {metadata_out}")
    print(f"provenance: {provenance_out}")


if __name__ == "__main__":
    main()
