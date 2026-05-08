# kdflow/opd_local/divergence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class CandidatePoint:
    token_idx: int
    loss_value: float


@dataclass
class DivergenceResult:
    found: bool
    candidate_idx: Optional[int]
    prefix_start_idx: Optional[int]
    probe_gap: Optional[float]
    threshold: Optional[float]


def select_topk_loss_candidates(
    token_losses: torch.Tensor,
    k: int,
    min_separation: int,
) -> List[CandidatePoint]:
    """
    Select top-k high-loss token positions with minimum distance constraint.
    token_losses: shape [T]
    """
    assert token_losses.dim() == 1
    values, indices = torch.sort(token_losses, descending=True)

    selected: List[CandidatePoint] = []
    for loss_val, idx in zip(values.tolist(), indices.tolist()):
        if all(abs(idx - p.token_idx) >= min_separation for p in selected):
            selected.append(CandidatePoint(token_idx=idx, loss_value=loss_val))
        if len(selected) >= k:
            break

    # Important: sort by position from left to right, not by loss.
    selected.sort(key=lambda x: x.token_idx)
    return selected


def compute_probe_gap_on_suffix(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
) -> float:
    """
    Both tensors are shape [L], aligned on the same suffix tokens.
    gap = avg(log p_T - log p_S)
    """
    assert teacher_logprobs.shape == student_logprobs.shape
    gap = (teacher_logprobs - student_logprobs).mean()
    return float(gap.item())


def dynamic_threshold(
    gap_values: Sequence[float],
    alpha: float,
    eps: float = 1e-6,
) -> float:
    """
    Threshold = mean + alpha * std
    Note: if there are too few values, std may be small, so keep eps.
    """
    if len(gap_values) == 0:
        return float("inf")
    x = torch.tensor(list(gap_values), dtype=torch.float32)
    mu = x.mean()
    sigma = x.std(unbiased=False)
    return float((mu + alpha * torch.clamp(sigma, min=eps)).item())


def find_true_divergence_point(
    token_losses: torch.Tensor,
    *,
    k: int,
    min_separation: int,
    rollback_steps: int,
    alpha: float,
    eps: float,
    probe_fn,
) -> DivergenceResult:
    """
    probe_fn(prefix_start_idx, candidate_idx) -> gap_value

    probe_fn should:
      1) build rollback prefix
      2) run teacher cheap probe or equivalent teacher scoring
      3) score the original student suffix
      4) return scalar gap
    """
    candidates = select_topk_loss_candidates(
        token_losses=token_losses,
        k=k,
        min_separation=min_separation,
    )

    gap_history: List[float] = []

    for cand in candidates:
        prefix_start_idx = max(0, cand.token_idx - rollback_steps)
        gap = probe_fn(
            prefix_start_idx=prefix_start_idx,
            candidate_idx=cand.token_idx,
        )

        thr = dynamic_threshold(gap_history + [gap], alpha=alpha, eps=eps)

        # You may later change this rule:
        # e.g., compare to statistics from all candidate gaps
        # or use running trajectory-level stats from token gaps.
        if gap > thr:
            return DivergenceResult(
                found=True,
                candidate_idx=cand.token_idx,
                prefix_start_idx=prefix_start_idx,
                probe_gap=gap,
                threshold=thr,
            )

        gap_history.append(gap)

    return DivergenceResult(
        found=False,
        candidate_idx=None,
        prefix_start_idx=None,
        probe_gap=None,
        threshold=None,
    )