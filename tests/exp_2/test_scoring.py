import numpy as np
from src.exp_2.scoring import geometric_score, bayesian_score


def test_geometric_score_basic():
    freq = np.array([10.0, 5.0, 1.0])
    avg_kl = np.array([1.0, 2.0, 3.0])
    scores = geometric_score(freq, avg_kl, alpha=0.3, beta=0.7)

    norm_freq = freq / freq.max()
    norm_kl = avg_kl / avg_kl.max()
    expected = (norm_freq**0.3) * (norm_kl**0.7)
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_geometric_score_uniform():
    """All same frequency and KL => all same score."""
    freq = np.array([5.0, 5.0, 5.0])
    avg_kl = np.array([2.0, 2.0, 2.0])
    scores = geometric_score(freq, avg_kl)
    np.testing.assert_allclose(scores, [1.0, 1.0, 1.0], rtol=1e-5)


def test_bayesian_score_basic():
    freq = np.array([100.0, 10.0, 1.0])
    avg_kl = np.array([2.0, 5.0, 10.0])
    C = 10.0
    mu = 3.0
    scores = bayesian_score(freq, avg_kl, C=C, mu=mu)

    expected = np.array([
        (100 * 2.0 + 10 * 3.0) / (100 + 10),  # 2.0909
        (10 * 5.0 + 10 * 3.0) / (10 + 10),     # 4.0
        (1 * 10.0 + 10 * 3.0) / (1 + 10),      # 3.6364
    ])
    np.testing.assert_allclose(scores, expected, rtol=1e-5)


def test_bayesian_score_low_freq_collapses_to_mu():
    """Very rare token's score should approach mu."""
    freq = np.array([0.001])
    avg_kl = np.array([100.0])
    C = 10.0
    mu = 3.0
    scores = bayesian_score(freq, avg_kl, C=C, mu=mu)
    np.testing.assert_allclose(scores, [mu], atol=0.1)


def test_bayesian_score_high_freq_keeps_true_kl():
    """Very frequent token's score should approach its true avg KL."""
    freq = np.array([100000.0])
    avg_kl = np.array([7.5])
    C = 10.0
    mu = 3.0
    scores = bayesian_score(freq, avg_kl, C=C, mu=mu)
    np.testing.assert_allclose(scores, [7.5], atol=0.01)
