from __future__ import annotations

import argparse
import csv
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.utils.checkpoint import file_sha256, git_commit, save_encoder_checkpoint, write_json

PINNED_STGNF_COMMIT = "edb5f3220332e160e4d20ce258787d5e2d7e0200"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    # Dataset adapter only: the official Graph implementation already includes the
    # NTU RGB+D 25-joint topology, while model_pose constructs Graph without exposing
    # the layout argument. Redirect that constructor to the official NTU layout.
    class NTUGraph(OfficialGraph):
        def __init__(self, strategy="spatial", headless=False, max_hop=1, **kwargs):
            super().__init__(layout="ntu-rgb+d", strategy=strategy, headless=headless, max_hop=max_hop)

    model_pose.Graph = NTUGraph
    return model_pose.STG_NF


def collate_xy(batch):
    # STG-NF is a 2-D human-pose anomaly detector. NTU provides XYZ skeletons;
    # project to the global X-Y plane so the vertical Y coordinate (fall signal)
    # is retained without modifying the published flow architecture.
    x = torch.stack([item["x"] for item in batch], dim=0)  # [N,T,V,3]
    x = x[..., :2].permute(0, 3, 1, 2).contiguous()  # [N,2,T,V]
    return x


def nll_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total = 0
    total_nll = 0.0
    for x in loader:
        x = x.to(device, non_blocking=True).float()
        if train:
            optimizer.zero_grad(set_to_none=True)
        _, nll = model(x, label=None)
        loss = nll.mean()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
        n = x.shape[0]
        total += n
        total_nll += float(loss.detach().item()) * n
    return total_nll / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train official STG-NF under the project's NTU120 normal-only protocol")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--official-repo", default="third_party/stgnf_official")
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/checkpoints/stgnf_official_normalonly.pt")
    ap.add_argument("--history-out", default=None)
    ap.add_argument("--run-metadata-out", default=None)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--preprocess-mode", choices=["frame_root", "sequence_origin"], default="sequence_origin")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--lr-decay", type=float, default=0.99)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=0)
    ap.add_argument("--R", type=float, default=0.0, help="0 keeps the unsupervised normal-only prior")
    ap.add_argument("--flow-permutation", default="permute")
    ap.add_argument("--adj-strategy", default="uniform")
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--temporal-kernel", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    seed_everything(args.seed)
    repo = Path(args.official_repo)
    STG_NF = import_stgnf(repo)
    commit = official_commit(repo)
    if commit != PINNED_STGNF_COMMIT:
        print(f"WARNING: official STG-NF commit is {commit}, expected pinned {PINNED_STGNF_COMMIT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NTUManifestDataset(
        args.manifest, args.root, inner_split="encoder_train", roles=["global_normal"],
        seq_len=args.seq_len, preprocess_mode=args.preprocess_mode,
    )
    val_ds = NTUManifestDataset(
        args.manifest, args.root, inner_split="encoder_val", roles=["global_normal"],
        seq_len=args.seq_len, preprocess_mode=args.preprocess_mode,
    )
    kwargs = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_xy)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = STG_NF(
        pose_shape=(2, args.seq_len, 25),
        hidden_channels=args.hidden_dim,
        K=args.K,
        L=args.L,
        actnorm_scale=1.0,
        flow_permutation=args.flow_permutation,
        flow_coupling="affine",
        LU_decomposed=True,
        learn_top=False,
        R=args.R,
        edge_importance=False,
        temporal_kernel_size=args.temporal_kernel,
        strategy=args.adj_strategy,
        max_hops=args.max_hops,
        device=str(device),
    ).to(device)

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history_out = Path(args.history_out) if args.history_out else out.with_suffix(".history.csv")
    metadata_out = Path(args.run_metadata_out) if args.run_metadata_out else out.with_suffix(".run.json")

    base_meta = {
        "detector_arch": "official_stgnf_iccv2023",
        "official_repo": "orhir/STG-NF",
        "official_commit": commit,
        "project_git_commit": git_commit(),
        "manifest_sha256": file_sha256(args.manifest),
        "normal_only_protocol": True,
        "train_role": "global_normal",
        "checkpoint_selection": "encoder_val global-normal mean NLL only",
        "ntu_projection": "XY_from_XYZ",
        "ntu_graph": "official Graph(layout='ntu-rgb+d')",
        "seq_len": args.seq_len,
        "preprocess_mode": args.preprocess_mode,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_decay": args.lr_decay,
        "K": args.K,
        "L": args.L,
        "hidden_dim": args.hidden_dim,
        "R": args.R,
        "flow_permutation": args.flow_permutation,
        "adj_strategy": args.adj_strategy,
        "max_hops": args.max_hops,
        "temporal_kernel": args.temporal_kernel,
        "seed": args.seed,
        "encoder_train_samples": len(train_ds),
        "encoder_val_samples": len(val_ds),
        "torch_version": str(torch.__version__),
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": str(device),
    }

    print(f"device: {device}")
    print(f"official_stgnf_commit: {commit}")
    print(f"train_samples: {len(train_ds)}")
    print(f"val_samples: {len(val_ds)}")
    print("input_adapter: NTU XYZ -> global XY projection; official NTU 25-joint graph")
    print("IMPORTANT: A42/A43 are not used for training or checkpoint selection.")

    best_val_nll = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        for g in optimizer.param_groups:
            g["lr"] = args.lr * (args.lr_decay ** (epoch - 1))
        train_nll = nll_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_nll = nll_epoch(model, val_loader, device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_nll": train_nll, "val_nll": val_nll}
        history.append(row)
        print(f"epoch {epoch:03d}/{args.epochs} lr={row['lr']:.6g} train_nll={train_nll:.6f} val_nll={val_nll:.6f}")
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            save_encoder_checkpoint(out, model.state_dict(), {**base_meta, "best_epoch": best_epoch, "best_val_nll": best_val_nll})
            print(f"saved_best: {out} val_nll={best_val_nll:.6f}")

    history_out.parent.mkdir(parents=True, exist_ok=True)
    with history_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    write_json(metadata_out, {
        **base_meta,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "checkpoint": str(out),
        "checkpoint_sha256": file_sha256(out),
        "history": str(history_out),
    })
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_nll: {best_val_nll:.6f}")
    print(f"checkpoint: {out}")
    print(f"history: {history_out}")
    print(f"run_metadata: {metadata_out}")


if __name__ == "__main__":
    main()
