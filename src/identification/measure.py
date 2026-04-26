"""Phase 2: Compute per-token KL divergence at both student checkpoints.

Helper functions for KL and entropy are at module level (testable without GPU).
The full Phase 2 orchestration (run_phase2) loads models and is GPU-only.
"""

import torch


# ---------------------------------------------------------------------------
# Pure-math helpers (testable on CPU)
# ---------------------------------------------------------------------------

def compute_kl_per_token(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(teacher || student) per token position.

    Args:
        teacher_logits: (seq_len, vocab_size) raw logits from teacher
        student_logits: (seq_len, vocab_size) raw logits from student

    Returns:
        (seq_len,) tensor of KL divergence values per position.
    """
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl


def compute_entropy_per_token(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the distribution at each position.

    Args:
        logits: (seq_len, vocab_size) raw logits

    Returns:
        (seq_len,) tensor of entropy values.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
