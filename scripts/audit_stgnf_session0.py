from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.utils.checkpoint import file_sha256, git_commit, load_encoder_checkpoint, write_json

PINNED_STGNF_COMMIT = "edb5f3220332e160e4d20ce258787d5e2d7e0200"


def official_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def import_stgnf(repo: Path):
    if not repo.exists():
        raise FileNotFoundError(f"Official STG-NF repo not found at {repo}. Run scripts/setup_stgnf_official.sh first.")
    sys.path.insert(0, str(repo.resolve()))
    import models.STG_NF.model_pose as model_pose
    from models.STG_NF.graph import Graph as OfficialGraph

    class NTUGraph(OfficialGraph):
        def __init__(self, strategy="spatial", headless=False, max_hop=1, **kwargs):
            super().__init__(layout="ntu-rgb+d", strategy=strategy, headless=headless, max_hop=max_hop)

    model_pose.Graph = NTUGraph
    return model_pose.STG_NF


def collate_xy_meta(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    x = x[..., :2].permute(0, 3, 1, 2).contiguous()  # [N,2,T,V]
    meta = [{k: v for k, v in item.items() if k != "x"} for item in batch]
    return x, meta


def build_model(STG_NF, meta, device):
    return STG_NF(
        pose_shape=(2, int(meta["seq_len"]), 25),
        hidden_channels=int(meta["hidden_dim"]),
        K=int(meta["K"]),
        L=int(meta["L"]),
        actnorm_scale=1.0,
        flow_permutation=str(meta["flow_permutation"]),
        flow_coupling="affine",
        LU_decomposed=True,
        learn_top=False,
        R=float(meta.get("R", 0.0)),
        edge_importance=False,
        temporal_kernel_size=meta.get("temporal_kernel"),
        strategy=str(meta["adj_strategy"]),
        max_hops=int(meta["max_hops"]),
        device=str(device),
    ).to(device)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Session-0 audit of official STG-NF on the locked NTU120 protocol")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--official-repo", default="third_party/stgnf_official")
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/session0/stgnf_session0.csv")
    ap.add_argument("--scores-out", default="outputs/ntu120_pilot_v0.1/session0/stgnf_scores.csv")
    ap.add_argument("--provenance-out", default="outputs/ntu120_pilot_v0.1/session0/stgnf_session0.provenance.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, ckpt_meta = load_encoder_checkpoint(args.checkpoint, map_location=device)
    if ckpt_meta.get("detector_arch") != "official_stgnf_iccv2023":
        raise RuntimeError(f"checkpoint is not an official STG-NF adapter checkpoint: {ckpt_meta.get('detector_arch')!r}")

    repo = Path(args.official_repo)
    STG_NF = import_stgnf(repo)
    commit = official_commit(repo)
    if ckpt_meta.get("official_commit") not in (None, commit):
        raise RuntimeError(f"official STG-NF commit mismatch: checkpoint={ckpt_meta.get('official_commit')} local={commit}")
    if commit != PINNED_STGNF_COMMIT:
        print(f"WARNING: local STG-NF commit is {commit}, expected pinned {PINNED_STGNF_COMMIT}")

    model = build_model(STG_NF, ckpt_meta, device)
    model.load_state_dict(state, strict=True)
    model.set_actnorm_init()
    model.eval()

    ds = NTUManifestDataset(
        args.manifest, args.root,
        roles=["global_normal", "candidate_personal_normal", "protected_anomaly"],
        seq_len=int(ckpt_meta["seq_len"]),
        preprocess_mode=str(ckpt_meta["preprocess_mode"]),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_xy_meta,
    )

    score_rows = []
    for bi, (x, batch_meta) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True).float()
        _, nll = model(x, label=None)
        scores = nll.detach().cpu().numpy().reshape(-1)
        for m, s in zip(batch_meta, scores):
            score_rows.append({**m, "score": float(s)})
        if bi % 50 == 0:
            print(f"batches: {bi}/{len(loader)}")

    def values(inner_split, role):
        return np.asarray([
            r["score"] for r in score_rows
            if r["inner_split"] == inner_split and r["role"] == role
        ], dtype=np.float64)

    train = values("encoder_train", "global_normal")
    calib = values("detector_calib", "global_normal")
    ret = values("retention_val", "global_normal")
    personal = values("deployment_test", "candidate_personal_normal")
    safe = values("deployment_test", "protected_anomaly")
    if min(len(train), len(calib), len(ret), len(personal), len(safe)) == 0:
        raise RuntimeError("one or more required protocol partitions are empty")

    threshold = float(np.quantile(calib, args.threshold_quantile))
    ret_fpr = float(np.mean(ret > threshold))
    personal_fpr = float(np.mean(personal > threshold))
    safe_recall = float(np.mean(safe > threshold))
    personal_mean = float(personal.mean())
    safe_mean = float(safe.mean())
    margin = safe_mean - personal_mean

    result = {
        "detector": "official_stgnf",
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "retFPR": ret_fpr,
        "persFPR": personal_fpr,
        "safeRec": safe_recall,
        "safeFNR": 1.0 - safe_recall,
        "personal_score_mean": personal_mean,
        "safe_score_mean": safe_mean,
        "margin": margin,
        "train_global_normal": len(train),
        "calib_global_normal": len(calib),
        "retention_global_normal": len(ret),
        "deployment_personal_normal": len(personal),
        "deployment_protected_anomaly": len(safe),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader(); w.writerow(result)

    scores_out = Path(args.scores_out)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "action", "setup", "camera", "repetition", "outer_split", "inner_split", "role", "path", "score"]
    with scores_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(score_rows)

    write_json(args.provenance_out, {
        "detector_arch": "official_stgnf_iccv2023",
        "official_repo": "orhir/STG-NF",
        "official_commit": commit,
        "project_git_commit_at_audit": git_commit(),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_metadata": ckpt_meta,
        "scores": str(scores_out),
        "scores_sha256": file_sha256(scores_out),
        "result": result,
        "protected_anomaly_policy": "A43 is evaluation-only; no model or threshold selection uses A43",
    })

    print(f"train_global_normal: {len(train)}")
    print(f"calib_global_normal: {len(calib)}")
    print(f"retention_global_normal: {len(ret)}")
    print(f"deployment_personal_normal: {len(personal)}")
    print(f"deployment_protected_anomaly: {len(safe)}")
    print()
    print("detector          retFPR  persFPR  safeRec     margin")
    print(f"official_stgnf    {ret_fpr:7.4f}  {personal_fpr:7.4f}  {safe_recall:7.4f}  {margin:9.6f}")
    print()
    print("IMPORTANT: A43 is evaluation-only. Do not tune STG-NF or threshold on protected-anomaly recall.")
    print(f"output: {out}")
    print(f"scores: {scores_out}")
    print(f"provenance: {args.provenance_out}")


if __name__ == "__main__":
    main()
