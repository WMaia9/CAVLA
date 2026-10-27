"""Standard U-statistics for responses of stochastic policies.

These estimators are established statistics, not a claimed research novelty.
Paired rows must be independent across seeds; dependence within a seed is allowed.
"""
import numpy as np


def paired_mean_effect_squared(original, edited):
    """Unbiased estimate of ||E[edited-original]||² / dimension.

The samplewise squared effect adds a variance term. The U-statistic removes it
in expectation. Negative estimates are possible and must not be clipped when
doing inference. This measures mean shift, not full distributional change.
"""
    a, b = np.asarray(original, dtype=np.float64), np.asarray(edited, dtype=np.float64)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("Need matching shapes and at least two independent seed pairs")
    delta = (b-a).reshape(len(a), -1)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite input")
    n, dimension = delta.shape
    total = np.sum(delta, axis=0)
    diagonal = np.sum(delta*delta)
    u_stat = (np.dot(total, total)-diagonal)/(n*(n-1)*dimension)
    return {"mean_effect_squared_unbiased": float(u_stat),
            "samplewise_effect_squared": float(diagonal/(n*dimension)),
            "squared_sample_mean_effect": float(np.mean(np.mean(delta, axis=0)**2)),
            "n_independent_seed_pairs": n}


def paired_rbf_mmd_squared(original, edited, bandwidth):
    """Unbiased squared MMD of marginals; excludes dependent cross diagonals.

Uses independent pair indices i != j in every term. The paired cross-diagonal
must not be included as if those draws were independent between conditions.
"""
    a, b = np.asarray(original, dtype=np.float64), np.asarray(edited, dtype=np.float64)
    if a.shape != b.shape or len(a) < 2 or bandwidth <= 0:
        raise ValueError("Need matching shapes, at least two pairs, positive bandwidth")
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite input")
    def kernel(x, y):
        distance = np.mean((x[:, None]-y[None, :])**2, axis=-1)
        return np.exp(-distance/(2*bandwidth**2))
    n = len(a)
    off_diag = ~np.eye(n, dtype=bool)
    return float((kernel(a, a)+kernel(b, b)-kernel(a, b)-kernel(b, a))[off_diag].mean())
