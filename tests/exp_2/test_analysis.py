# tests/exp_2/test_analysis.py
import numpy as np
import torch

from src.exp_2.phases import aggregate_token_stats, classify_tokens


def test_classify_tokens_below_threshold_is_pillar():
    avg_entropies = np.array([1.0, 3.0, 2.5])
    threshold = 2.5
    labels = classify_tokens(avg_entropies, threshold)
    assert labels == ["pillar", "stumbling_block", "stumbling_block"]


def test_classify_tokens_all_pillar():
    avg_entropies = np.array([0.5, 1.0, 1.5])
    threshold = 5.0
    labels = classify_tokens(avg_entropies, threshold)
    assert labels == ["pillar", "pillar", "pillar"]


def test_aggregate_token_stats():
    # Two samples: tokens [10, 20, 10] and [20, 30, 10]
    all_token_ids = torch.tensor([10, 20, 10, 20, 30, 10])
    all_kl = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    all_entropy = torch.tensor([0.5, 1.0, 0.7, 1.2, 2.0, 0.9])

    unique_ids, freqs, avg_kls, avg_entropies = aggregate_token_stats(
        all_token_ids, all_kl, all_entropy
    )

    # Token 10: freq=3, avg_kl=(1+3+6)/3=10/3, avg_entropy=(0.5+0.7+0.9)/3
    # Token 20: freq=2, avg_kl=(2+4)/2=3, avg_entropy=(1.0+1.2)/2
    # Token 30: freq=1, avg_kl=5, avg_entropy=2.0
    idx_10 = (unique_ids == 10).nonzero().item()
    idx_20 = (unique_ids == 20).nonzero().item()
    idx_30 = (unique_ids == 30).nonzero().item()

    assert freqs[idx_10] == 3
    assert freqs[idx_20] == 2
    assert freqs[idx_30] == 1

    np.testing.assert_allclose(avg_kls[idx_10], 10 / 3, rtol=1e-5)
    np.testing.assert_allclose(avg_kls[idx_20], 3.0, rtol=1e-5)
    np.testing.assert_allclose(avg_kls[idx_30], 5.0, rtol=1e-5)
