import torch

from src.identification.identify import classify_rock_tokens, aggregate_to_types


def test_classify_high_before_low_improvement_is_rock():
    """Token with high loss_before and barely any improvement → Rock."""
    loss_before = torch.tensor([10.0, 10.0, 1.0, 1.0, 10.0])
    loss_after = torch.tensor([9.5, 2.0, 0.5, 0.8, 9.8])
    is_rock = classify_rock_tokens(loss_before, loss_after, tau_high_pct=60, delta_pct=40)
    assert is_rock[4].item() is True
    assert is_rock[1].item() is False  # high improvement (Learned)
    assert is_rock[2].item() is False  # low loss_before (Easy)


def test_classify_all_easy():
    """All tokens with low loss → none are Rock."""
    loss_before = torch.tensor([0.1, 0.2, 0.3])
    loss_after = torch.tensor([0.05, 0.1, 0.15])
    is_rock = classify_rock_tokens(loss_before, loss_after, tau_high_pct=80, delta_pct=20)
    assert is_rock.sum().item() <= 1


def test_aggregate_frequency_filter():
    """Tokens appearing fewer than min_frequency times are excluded."""
    token_ids = torch.tensor([1, 1, 1, 2, 2, 2, 2, 2, 3])
    is_rock = torch.tensor([True, True, True, True, False, False, False, False, True])
    result = aggregate_to_types(token_ids, is_rock, min_frequency=5, top_k=10)
    assert len(result) == 1
    assert result[0]["token_id"] == 2
    assert abs(result[0]["rock_rate"] - 0.2) < 1e-6


def test_aggregate_top_k_ranking():
    """Top-K returns tokens sorted by rock_rate descending."""
    token_ids = torch.tensor([1] * 20 + [2] * 20 + [3] * 20)
    is_rock = torch.tensor(
        [True] * 18 + [False] * 2
        + [True] * 10 + [False] * 10
        + [True] * 4 + [False] * 16
    )
    result = aggregate_to_types(token_ids, is_rock, min_frequency=10, top_k=2)
    assert len(result) == 2
    assert result[0]["token_id"] == 1
    assert result[1]["token_id"] == 2
