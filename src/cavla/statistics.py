"""Scene-clustered statistics and train-only normalization."""
import numpy as np
from scipy.stats import spearmanr


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return None
    return float(spearmanr(x[valid], y[valid]).statistic)


def grouped_folds(groups, n_folds=5, seed=20261020):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < n_folds:
        raise ValueError("Need at least one independent group per fold")
    unique = np.random.default_rng(seed).permutation(unique)
    for held_out in np.array_split(unique, n_folds):
        test = np.isin(groups, held_out)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def cluster_bootstrap(values, groups, repeats=1000, seed=20261020):
    """Equal weight per scene, then bootstrap independent scenes."""
    values, groups = np.asarray(values), np.asarray(groups)
    unique = np.unique(groups)
    means = np.array([np.mean(values[groups == g]) for g in unique])
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(means, size=(repeats, len(means))), axis=1)
    return {"mean": float(means.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "n_independent_scenes": len(unique)}


def ridge_predict(x_train, y_train, x_test, alpha=10.0):
    """Median imputation, scaling and ridge fitted exclusively on training rows."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.
                       for c in x_train.T])
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    mean, std = x_train.mean(0), x_train.std(0)
    std = np.where(std > 1e-8, std, 1.)
    a, b = (x_train - mean) / std, (x_test - mean) / std
    target_mean = np.mean(y_train, axis=0)
    weights = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]),
                              a.T @ (y_train - target_mean))
    return b @ weights + target_mean
