from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def save_encoder_checkpoint(
    path: str | Path,
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "state_dict": state_dict, "metadata": metadata}, path)


def load_encoder_checkpoint(path: str | Path, map_location=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(path, map_location=map_location)
    # Backward compatibility with the earlier raw-state-dict checkpoints.
    if isinstance(obj, dict) and "state_dict" in obj and "metadata" in obj:
        return obj["state_dict"], dict(obj.get("metadata") or {})
    if isinstance(obj, dict):
        return obj, {"legacy_raw_state_dict": True}
    raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
