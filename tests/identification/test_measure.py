import torch

from src.identification.measure import (
    compute_kl_per_token,
    compute_kl_from_cache,
    compute_entropy_per_token,
    _build_batches,
)


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


def test_kl_from_cache_matches_exact():
    """Top-K cached KL should closely match exact KL with peaked distributions."""
    torch.manual_seed(42)
    # Scale up logits to make distribution peaked (like real LMs)
    teacher_logits = torch.randn(5, 100) * 5.0
    student_logits = torch.randn(5, 100) * 5.0

    exact_kl = compute_kl_per_token(teacher_logits, student_logits)

    # Build cache (top-50 of 100 — with peaked distributions, captures >99% of mass)
    log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    top_log_probs, top_indices = log_probs.topk(50, dim=-1)
    cache = {"top_indices": top_indices, "top_log_probs": top_log_probs}

    cached_kl = compute_kl_from_cache(cache, student_logits)

    assert cached_kl.shape == exact_kl.shape
    assert torch.allclose(cached_kl, exact_kl, atol=0.01), (
        f"Max diff: {(cached_kl - exact_kl).abs().max()}"
    )


def test_build_batches_respects_token_limit():
    """Batches should not exceed max_batch_tokens."""
    sequences = [
        {"prompt_token_ids": [0] * 100, "output_token_ids": [0] * 900},  # 1000
        {"prompt_token_ids": [0] * 100, "output_token_ids": [0] * 900},  # 1000
        {"prompt_token_ids": [0] * 100, "output_token_ids": [0] * 900},  # 1000
        {"prompt_token_ids": [0] * 500, "output_token_ids": [0] * 4500}, # 5000
    ]
    batches = _build_batches(sequences, max_batch_tokens=3000)
    # First 3 sequences (1000 each) can fit in one batch: 3 × 1000 = 3000
    # Fourth (5000) must be alone
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 1
