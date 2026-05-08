# kdflow/opd_local/margin.py
from __future__ import annotations

from typing import List, Dict

import torch


def compute_trajectory_margin(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
) -> float:
    """
    teacher_logprobs, student_logprobs: shape [L]
    margin = avg(log p_T - log p_S)
    """
    assert teacher_logprobs.shape == student_logprobs.shape
    margin = (teacher_logprobs - student_logprobs).mean()
    return float(margin.item())


def compute_margin_for_trajectory(
    *,
    score_fn,
    prefix_input_ids,
    continuation_token_ids,
) -> float:
    """
    score_fn(prefix_input_ids, continuation_token_ids) -> {
        "teacher_logprobs": Tensor[L],
        "student_logprobs": Tensor[L],
    }
    """
    out = score_fn(
        prefix_input_ids=prefix_input_ids,
        continuation_token_ids=continuation_token_ids,
    )
    return compute_trajectory_margin(
        teacher_logprobs=out["teacher_logprobs"],
        student_logprobs=out["student_logprobs"],
    )


def batch_compute_margins(
    *,
    score_fn,
    prefix_input_ids,
    trajectories: List[Dict],
) -> List[float]:
    margins: List[float] = []
    for traj in trajectories:
        m = compute_margin_for_trajectory(
            score_fn=score_fn,
            prefix_input_ids=prefix_input_ids,
            continuation_token_ids=traj["token_ids"],
        )
        margins.append(m)
    return margins