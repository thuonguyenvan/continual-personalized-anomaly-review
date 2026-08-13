from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.manifest_dataset import NTUManifestDataset
from src.models.simple_skeleton_encoder import SimpleSkeletonEncoder
from src.utils.checkpoint import file_sha256, git_commit, save_encoder_checkpoint, write_json

GLOBAL_NORMAL_ACTIONS = [1, 2, 6, 8, 9, 11, 12, 23, 25, 28, 29, 33, 34, 35, 36, 37, 40, 80, 96, 97, 98, 99, 100, 101]
ACTION_TO_CLASS = {a: i for i, a in enumerate(GLOBAL_NORMAL_ACTIONS)}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collate(batch):
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.tensor([ACTION_TO_CLASS[item["action"]] for item in batch], dtype=torch.long)
    return x, y


class ActionClassifier(nn.Module):
    def __init__(self, embed_dim: int = 128, num_classes: int = len(GLOBAL_NORMAL_ACTIONS)):
        super().__init__(); self.encoder = SimpleSkeletonEncoder(embed_dim=embed_dim); self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.head(self.encoder(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None; model.train(is_train)
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train: optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = criterion(logits, y)
        if is_train: loss.backward(); optimizer.step()
        total_loss += float(loss.item()) * y.size(0); correct += int((logits.argmax(1) == y).sum().item()); total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NTU120 skeleton encoder without reusing detector-calibration subjects")
    ap.add_argument("--manifest", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="checkpoints/simple_skeleton_encoder.pt")
    ap.add_argument("--history-out", default=None); ap.add_argument("--run-metadata-out", default=None)
    ap.add_argument("--seq-len", type=int, default=64); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0); args = ap.parse_args()

    seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NTUManifestDataset(args.manifest, args.root, inner_split="encoder_train", roles=["global_normal"], seq_len=args.seq_len)
    val_ds = NTUManifestDataset(args.manifest, args.root, inner_split="encoder_val", roles=["global_normal"], seq_len=args.seq_len)
    if len(train_ds) == 0 or len(val_ds) == 0: raise RuntimeError("Empty encoder_train/encoder_val split. Regenerate the manifest.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = ActionClassifier().to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    history_out = Path(args.history_out) if args.history_out else out.with_suffix(".history.csv")
    metadata_out = Path(args.run_metadata_out) if args.run_metadata_out else out.with_suffix(".run.json")
    best_val_acc = -1.0; best_epoch = -1; history = []
    manifest_hash = file_sha256(args.manifest); commit = git_commit()
    base_metadata = {"git_commit": commit, "manifest_sha256": manifest_hash, "seed": args.seed, "seq_len": args.seq_len, "batch_size": args.batch_size, "epochs_requested": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay, "global_normal_actions": GLOBAL_NORMAL_ACTIONS, "encoder_train_samples": len(train_ds), "encoder_val_samples": len(val_ds), "torch_version": str(torch.__version__), "python_version": sys.version, "platform": platform.platform(), "device": str(device)}

    print(f"device: {device}"); print(f"train_samples: {len(train_ds)}"); print(f"val_samples: {len(val_ds)}"); print(f"num_classes: {len(GLOBAL_NORMAL_ACTIONS)}"); print(f"git_commit: {commit}"); print(f"manifest_sha256: {manifest_hash}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad(): val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            save_encoder_checkpoint(out, model.encoder.state_dict(), {**base_metadata, "best_epoch": best_epoch, "best_val_acc": best_val_acc})
            print(f"saved_best_encoder: {out} (val_acc={best_val_acc:.4f})")

    with history_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]); w.writeheader(); w.writerows(history)
    write_json(metadata_out, {**base_metadata, "best_epoch": best_epoch, "best_val_acc": best_val_acc, "checkpoint": str(out), "checkpoint_sha256": file_sha256(out), "history": str(history_out)})
    print(f"best_val_acc: {best_val_acc:.4f}"); print(f"best_epoch: {best_epoch}"); print(f"checkpoint: {out}"); print(f"history: {history_out}"); print(f"run_metadata: {metadata_out}")


if __name__ == "__main__": main()
