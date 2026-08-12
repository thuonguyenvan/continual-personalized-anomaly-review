from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.datasets.ntu120 import read_skeleton_file
from src.preprocessing.skeleton import preprocess_skeleton


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    subject: int
    action: int
    setup: int
    camera: int
    repetition: int
    outer_split: str
    inner_split: str
    role: str


class NTUManifestDataset(Dataset):
    """Dataset backed by the manifest produced by make_ntu120_manifest.py.

    Expected columns:
      path, setup, camera, subject, repetition, action,
      outer_split, inner_split, role
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path,
        inner_split: Optional[str] = None,
        outer_split: Optional[str] = None,
        roles: Optional[List[str]] = None,
        subject_ids: Optional[List[int]] = None,
        seq_len: int = 64,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir)
        self.seq_len = int(seq_len)
        self.transform = transform

        role_set = set(roles) if roles is not None else None
        subject_set = set(subject_ids) if subject_ids is not None else None

        records: List[ManifestRecord] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {
                "path", "setup", "camera", "subject", "repetition", "action",
                "outer_split", "inner_split", "role",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

            for row in reader:
                record = ManifestRecord(
                    path=row["path"],
                    setup=int(row["setup"]),
                    camera=int(row["camera"]),
                    subject=int(row["subject"]),
                    repetition=int(row["repetition"]),
                    action=int(row["action"]),
                    outer_split=row["outer_split"],
                    inner_split=row["inner_split"],
                    role=row["role"],
                )
                if inner_split is not None and record.inner_split != inner_split:
                    continue
                if outer_split is not None and record.outer_split != outer_split:
                    continue
                if role_set is not None and record.role not in role_set:
                    continue
                if subject_set is not None and record.subject not in subject_set:
                    continue
                records.append(record)

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, record_path: str) -> Path:
        p = Path(record_path)
        if p.is_absolute():
            return p
        return self.root_dir / p

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        skeleton_path = self._resolve_path(record.path)
        raw = read_skeleton_file(skeleton_path)
        sequence = preprocess_skeleton(raw, target_len=self.seq_len)
        x = torch.as_tensor(sequence, dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x)

        return {
            "x": x,
            "subject": record.subject,
            "action": record.action,
            "setup": record.setup,
            "camera": record.camera,
            "repetition": record.repetition,
            "outer_split": record.outer_split,
            "inner_split": record.inner_split,
            "role": record.role,
            "path": record.path,
        }
