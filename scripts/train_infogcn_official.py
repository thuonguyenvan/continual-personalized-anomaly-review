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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.utils.checkpoint import file_sha256, git_commit, save_encoder_checkpoint, write_json

GLOBAL_NORMAL_ACTIONS = [1, 2, 6, 8, 9, 11, 12, 23, 25, 28, 29, 33, 34, 35, 36, 37, 40, 80, 96, 97, 98, 99, 100, 101]
ACTION_TO_CLASS = {a: i for i, a in enumerate(GLOBAL_NORMAL_ACTIONS)}
PINNED_INFOGCN_COMMIT = "873feaa85160317335a83e04013e0ffa3f63525e"


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


def import_infogcn(repo: Path):
    if not repo.exists():
        raise FileNotFoundError(f"Official InfoGCN repo not found at {repo}. Run scripts/setup_infogcn_official.sh first.")
    sys.path.insert(0, str(repo.resolve()))
    from model.infogcn import InfoGCN
    return InfoGCN


def collate(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)  # [N,T,V,C]
    x = x.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()  # [N,C,T,V,M=1]
    y = torch.tensor([ACTION_TO_CLASS[item["action"]] for item in batch], dtype=torch.long)
    return x, y


def infogcn_loss(logits, z, y, z_prior, lambda_1: float, lambda_2: float):
    cls = F.cross_entropy(logits, y)
    present = torch.unique(y)
    means = torch.stack([z[y == c].mean(dim=0) for c in present], dim=0)
    # Official InfoGCN keeps z_prior as a plain CPU tensor rather than a registered
    # parameter/buffer, so model.to(device) does not move it. Move the prior first,
    # then index it with the CUDA class ids to avoid CPU/CUDA indexing mismatch.
    prior = z_prior.to(device=z.device, dtype=z.dtype)[present]
    mmd = F.mse_loss(means, prior)
    l2 = torch.linalg.vector_norm(z.mean(dim=0), ord=2)
    return cls + lambda_2 * mmd + lambda_1 * l2, cls, mmd, l2


def run_epoch(model, loader, device, lambda_1, lambda_2, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total = 0
    correct = 0
    sums = np.zeros(4, dtype=np.float64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits, z = model(x)
        loss, cls, mmd, l2 = infogcn_loss(logits, z, y, model.z_prior, lambda_1, lambda_2)
        if train:
            loss.backward()
            optimizer.step()
        n = y.size(0)
        vals = [loss, cls, mmd, l2]
        for i, v in enumerate(vals):
            sums[i] += float(v.detach().item()) * n
        correct += int((logits.argmax(1) == y).sum().item())
        total += n
    return {
        "loss": sums[0] / max(total, 1),
        "cls_loss": sums[1] / max(total, 1),
        "mmd_loss": sums[2] / max(total, 1),
        "l2_loss": sums[3] / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the official InfoGCN architecture under the project's normal-only protocol")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--official-repo", default="third_party/infogcn_official")
    ap.add_argument("--out", default="outputs/ntu120_pilot_v0.1/checkpoints/infogcn_official_normalonly.pt")
    ap.add_argument("--history-out", default=None)
    ap.add_argument("--run-metadata-out", default=None)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--preprocess-mode", choices=["frame_root", "sequence_origin"], default="sequence_origin")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--base-lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--steps", nargs="+", type=int, default=[90, 100])
    ap.add_argument("--lr-decay", type=float, default=0.1)
    ap.add_argument("--lambda-1", type=float, default=1e-4)
    ap.add_argument("--lambda-2", type=float, default=1e-1)
    ap.add_argument("--noise-ratio", type=float, default=0.5)
    ap.add_argument("--num-head", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--z-prior-gain", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    seed_everything(args.seed)
    repo = Path(args.official_repo)
    InfoGCN = import_infogcn(repo)
    commit = official_commit(repo)
    if commit != PINNED_INFOGCN_COMMIT:
        print(f"WARNING: official InfoGCN commit is {commit}, expected pinned {PINNED_INFOGCN_COMMIT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NTUManifestDataset(args.manifest, args.root, inner_split="encoder_train", roles=["global_normal"], seq_len=args.seq_len, preprocess_mode=args.preprocess_mode)
    val_ds = NTUManifestDataset(args.manifest, args.root, inner_split="encoder_val", roles=["global_normal"], seq_len=args.seq_len, preprocess_mode=args.preprocess_mode)
    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = InfoGCN(
        num_class=len(GLOBAL_NORMAL_ACTIONS), num_point=25, num_person=1,
        graph="graph.ntu_rgb_d.Graph", in_channels=3, drop_out=0,
        num_head=args.num_head, noise_ratio=args.noise_ratio, k=args.k, gain=args.z_prior_gain,
    ).to(device)
    model.A_vector = model.A_vector.to(device=device, dtype=torch.float32)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history_out = Path(args.history_out) if args.history_out else out.with_suffix(".history.csv")
    metadata_out = Path(args.run_metadata_out) if args.run_metadata_out else out.with_suffix(".run.json")

    base_meta = {
        "encoder_arch": "official_infogcn_cvpr2022",
        "official_repo": "stnoah1/infogcn",
        "official_commit": commit,
        "project_git_commit": git_commit(),
        "manifest_sha256": file_sha256(args.manifest),
        "normal_only_protocol": True,
        "global_normal_actions": GLOBAL_NORMAL_ACTIONS,
        "num_classes": len(GLOBAL_NORMAL_ACTIONS),
        "num_person": 1,
        "seq_len": args.seq_len,
        "preprocess_mode": args.preprocess_mode,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "base_lr": args.base_lr,
        "steps": args.steps,
        "warmup_epochs": args.warmup_epochs,
        "lr_decay": args.lr_decay,
        "weight_decay": args.weight_decay,
        "lambda_1": args.lambda_1,
        "lambda_2": args.lambda_2,
        "noise_ratio": args.noise_ratio,
        "num_head": args.num_head,
        "k": args.k,
        "z_prior_gain": args.z_prior_gain,
        "seed": args.seed,
        "encoder_train_samples": len(train_ds),
        "encoder_val_samples": len(val_ds),
        "torch_version": str(torch.__version__),
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": str(device),
    }

    print(f"device: {device}")
    print(f"official_infogcn_commit: {commit}")
    print(f"train_samples: {len(train_ds)}")
    print(f"val_samples: {len(val_ds)}")
    print(f"A_vector_dtype: {model.A_vector.dtype}")
    print("IMPORTANT: A42/A43 are not used for training or checkpoint selection.")

    best_val_acc = -1.0
    best_epoch = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            lr = args.base_lr * epoch / max(args.warmup_epochs, 1)
        else:
            lr = args.base_lr * (args.lr_decay ** sum((epoch - 1) >= s for s in args.steps))
        for g in optimizer.param_groups:
            g["lr"] = lr

        tr = run_epoch(model, train_loader, device, args.lambda_1, args.lambda_2, optimizer)
        with torch.no_grad():
            va = run_epoch(model, val_loader, device, args.lambda_1, args.lambda_2)

        row = {"epoch": epoch, "lr": lr}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        history.append(row)
        print(f"epoch {epoch:03d}/{args.epochs} lr={lr:.5g} train_acc={tr['acc']:.4f} val_acc={va['acc']:.4f} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f}")

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            best_epoch = epoch
            save_encoder_checkpoint(out, model.state_dict(), {**base_meta, "best_epoch": best_epoch, "best_val_acc": best_val_acc})
            print(f"saved_best: {out} val_acc={best_val_acc:.4f}")

    history_out.parent.mkdir(parents=True, exist_ok=True)
    with history_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    write_json(metadata_out, {**base_meta, "best_epoch": best_epoch, "best_val_acc": best_val_acc, "checkpoint": str(out), "checkpoint_sha256": file_sha256(out), "history": str(history_out)})
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print(f"checkpoint: {out}")
    print(f"history: {history_out}")
    print(f"run_metadata: {metadata_out}")


if __name__ == "__main__":
    main()
