"""Utilities for NTU RGB+D 120 skeleton metadata and raw skeleton parsing.

This module deliberately keeps dataset semantics separate from the experiment
protocol. Action roles (global normal, candidate personal normal, protected
anomaly) belong in configuration files/manifests, not in this parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np


_FILENAME_RE = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R(?P<rep>\d{3})A(?P<action>\d{3})"
)


@dataclass(frozen=True)
class NTUMetadata:
    path: str
    setup: int
    camera: int
    subject: int
    repetition: int
    action: int


def parse_filename(path: str | Path) -> NTUMetadata:
    """Parse NTU sample metadata from a canonical filename."""
    p = Path(path)
    m = _FILENAME_RE.search(p.stem)
    if m is None:
        raise ValueError(f"Invalid NTU filename: {p.name}")
    g = {k: int(v) for k, v in m.groupdict().items()}
    return NTUMetadata(
        path=str(p),
        setup=g["setup"],
        camera=g["camera"],
        subject=g["subject"],
        repetition=g["rep"],
        action=g["action"],
    )


def scan_skeleton_files(root: str | Path) -> list[NTUMetadata]:
    root = Path(root)
    files = sorted(root.rglob("*.skeleton"))
    return [parse_filename(p) for p in files]


def read_skeleton_file(path: str | Path, max_bodies: int = 2, num_joints: int = 25) -> np.ndarray:
    """Read an NTU .skeleton file.

    Returns
    -------
    np.ndarray
        Shape (T, M, V, 3), where T=frames, M=max_bodies, V=num_joints.
        Missing bodies/joints are zero-padded.

    Notes
    -----
    Only XYZ coordinates are retained for the first mechanistic pilot.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        n_frames = int(f.readline())
        data = np.zeros((n_frames, max_bodies, num_joints, 3), dtype=np.float32)

        for t in range(n_frames):
            n_bodies = int(f.readline())
            for b in range(n_bodies):
                _body_info = f.readline()  # body metadata, not used in pilot
                n_joints = int(f.readline())
                for j in range(n_joints):
                    vals = f.readline().split()
                    if b < max_bodies and j < num_joints:
                        data[t, b, j, 0] = float(vals[0])
                        data[t, b, j, 1] = float(vals[1])
                        data[t, b, j, 2] = float(vals[2])
    return data


def metadata_to_rows(items: Iterable[NTUMetadata]) -> list[dict]:
    return [item.__dict__.copy() for item in items]
