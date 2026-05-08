# kdflow/opd_local/local_loss.py
from __future__ import annotations

from typing import List

import torch


def sorted_pair_distance(
    teacher_margins: List[float],
    student_margins: List[float],
    distance_type: str = "l1",
) -> torch.Tensor:
    """
    1D transport-style approximation:
    sort both scalar sets and match by rank.
    """
    t = torch.tensor(sorted(teacher_margins), dtype=torch.float32)
    s = torch.tensor(sorted(student_margins), dtype=torch.float32)

    n = min(t.numel(), s.numel())
    if n == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    t = t[:n]
    s = s[:n]

    if distance_type == "l1":
        return torch.abs(t - s).mean()
    if distance_type == "l2":
        return ((t - s) ** 2).mean()

    raise ValueError(f"Unsupported distance_type: {distance_type}")