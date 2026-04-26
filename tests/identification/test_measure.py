import torch

from src.identification.measure import compute_kl_per_token, compute_entropy_per_token


def test_kl_identical_distributions_is_zero():
    """KL(P || P) = 0 for any distribution."""
    logits = torch.randn(5, 100)  # 5 positions, vocab 100
    kl = compute_kl_per_token(logits, logits)
    assert kl.shape == (5,)
    assert torch.allclose(kl, torch.zeros(5), atol=1e-5)


def test_kl_is_nonnegative():
    """KL divergence is always >= 0."""
    teacher_logits = torch.randn(10, 50)
    student_logits = torch.randn(10, 50)
    kl = compute_kl_per_token(teacher_logits, student_logits)
    assert (kl >= -1e-6).all(), f"Negative KL found: {kl.min()}"


def test_kl_peaked_teacher_high_kl():
    """When teacher is peaked and student is uniform, KL should be high."""
    teacher_logits = torch.zeros(1, 10)
    teacher_logits[0, 0] = 100.0  # very peaked at token 0

    student_logits = torch.zeros(1, 10)  # uniform

    kl = compute_kl_per_token(teacher_logits, student_logits)
    assert kl[0] > 1.0  # should be large


def test_entropy_uniform_is_log_vocab():
    """Entropy of uniform distribution = log(vocab_size)."""
    import math
    vocab = 100
    logits = torch.zeros(3, vocab)  # uniform
    entropy = compute_entropy_per_token(logits)
    expected = math.log(vocab)
    assert torch.allclose(entropy, torch.full((3,), expected), atol=1e-4)


def test_entropy_peaked_is_near_zero():
    """Entropy of a peaked distribution is near 0."""
    logits = torch.full((2, 50), -1000.0)
    logits[:, 0] = 100.0  # all mass on token 0
    entropy = compute_entropy_per_token(logits)
    assert (entropy < 0.01).all()
