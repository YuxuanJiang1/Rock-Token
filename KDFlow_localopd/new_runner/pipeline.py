# kdflow/opd_local/pipeline.py
from __future__ import annotations

from typing import Dict, Optional

import torch

from .config import LocalOPDConfig
from .divergence import find_true_divergence_point
from .local_sampling import sample_local_continuations, build_student_local_set
from .margin import batch_compute_margins
from .local_loss import sorted_pair_distance


def compute_local_divergence_loss_for_trajectory(
    *,
    cfg: LocalOPDConfig,
    full_student_token_ids,
    token_losses: torch.Tensor,
    build_prefix_fn,
    extract_original_suffix_fn,
    probe_fn,
    teacher_sample_fn,
    student_sample_fn,
    score_fn,
) -> Dict:
    """
    Returns:
      {
        "found": bool,
        "local_loss": Tensor,
        "debug": {...}
      }
    """

    div_result = find_true_divergence_point(
        token_losses=token_losses,
        k=cfg.num_candidates,
        min_separation=cfg.min_candidate_separation,
        rollback_steps=cfg.rollback_steps,
        alpha=cfg.threshold_alpha,
        eps=cfg.threshold_eps,
        probe_fn=probe_fn,
    )

    if not div_result.found:
        return {
            "found": False,
            "local_loss": torch.tensor(0.0, dtype=torch.float32),
            "debug": {"divergence_result": div_result},
        }

    prefix_input_ids = build_prefix_fn(div_result.prefix_start_idx)

    teacher_trajs = sample_local_continuations(
        sample_fn=teacher_sample_fn,
        prefix_input_ids=prefix_input_ids,
        num_samples=cfg.teacher_num_samples,
        max_new_tokens=cfg.local_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    # Because one slot is reserved for the original suffix.
    num_new_student = max(cfg.student_num_samples - 1, 0)
    new_student_trajs = sample_local_continuations(
        sample_fn=student_sample_fn,
        prefix_input_ids=prefix_input_ids,
        num_samples=num_new_student,
        max_new_tokens=cfg.local_len,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    original_suffix = extract_original_suffix_fn(
        div_result.prefix_start_idx,
        cfg.local_len,
    )
    student_trajs = build_student_local_set(
        original_suffix_token_ids=original_suffix,
        newly_sampled_student=new_student_trajs,
    )

    teacher_margins = batch_compute_margins(
        score_fn=score_fn,
        prefix_input_ids=prefix_input_ids,
        trajectories=teacher_trajs,
    )
    student_margins = batch_compute_margins(
        score_fn=score_fn,
        prefix_input_ids=prefix_input_ids,
        trajectories=student_trajs,
    )

    local_loss = sorted_pair_distance(
        teacher_margins=teacher_margins,
        student_margins=student_margins,
        distance_type=cfg.distance_type,
    )

    return {
        "found": True,
        "local_loss": local_loss,
        "debug": {
            "divergence_result": div_result,
            "teacher_margins": teacher_margins,
            "student_margins": student_margins,
        },
    }