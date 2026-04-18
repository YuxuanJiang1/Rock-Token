import numpy as np


def geometric_score(
    frequencies: np.ndarray,
    avg_kls: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> np.ndarray:
    """Beta-weighted geometric mean Rock Score.

    RockScore_i = (f_i / max(f))^alpha * (avg_KL_i / max(avg_KL))^beta
    """
    norm_freq = frequencies / frequencies.max()
    norm_kl = avg_kls / avg_kls.max()
    return (norm_freq**alpha) * (norm_kl**beta)


def bayesian_score(
    frequencies: np.ndarray,
    avg_kls: np.ndarray,
    C: float,
    mu: float,
) -> np.ndarray:
    """Bayesian averaging (Laplace smoothing) Rock Score.

    BayesianKL_i = (f_i * avg_KL_i + C * mu) / (f_i + C)

    Args:
        C: confidence constant (e.g., median frequency across all token types)
        mu: global average KL across all token positions
    """
    return (frequencies * avg_kls + C * mu) / (frequencies + C)
