from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.datasets.ntu120 import read_skeleton_file
from src.preprocessing.skeleton import preprocess_skeleton_sequence


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    subject_id: int
    action_id: int
    setup_id: int
    camera_id: int
    repetition_id: int
    split: str
    role: str


class NTUManifestDataset(Dataset):
    """Dataset backed by a CSV manifest.

    Expected columns:
      path, subject_id, action_id, setup_id, camera_id, repetition_id, split, role

    `role` should be one of values such as:
      global_normal, personal_normal, protected_anomaly, excluded
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path,
        split: Optional[str] = None,
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
                "path",
                "subject_id",
                "action_id",
                "setup_id",
                "camera_id",
                "repetition_id",
                "split",
                "role",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

            for row in reader:
                record = ManifestRecord(
                    path=row["path"],
                    subject_id=int(row["subject_id"]),
                    action_id=int(row["action_id"]),
                    setup_id=int(row["setup_id"]),
                    camera_id=int(row["camera_id"]),
                    repetition_id=int(row["repetition_id"]),
                    split=row["split"],
                    role=row["role"],
                )
                if split is not None and record.split != split:
                    continue
                if role_set is not None and record.role not in role_set:
                    continue
                if subject_set is not None and record.subject_id not in subject_set:
                    continue
                records.append(record)

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        skeleton_path = self.root_dir / record.path
        raw = read_skeleton_file(skeleton_path)
        sequence = preprocess_skeleton_sequence(raw, target_len=self.seq_len)
        x = torch.as_tensor(sequence, dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x)

        return {
            "x": x,
            "subject_id": record.subject_id,
            "action_id": record.action_id,
            "setup_id": record.setup_id,
            "camera_id": record.camera_id,
            "repetition_id": record.repetition_id,
            "split": record.split,
            "role": record.role,
            "path": record.path,
        }
