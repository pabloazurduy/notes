from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _gaussian_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    variance = max(float(variance), 1e-6)
    coeff = 1.0 / np.sqrt(2.0 * np.pi * variance)
    exponent = np.exp(-0.5 * ((x - mean) ** 2) / variance)
    return coeff * exponent


def expectation_maximization_gaussian_mixture(
    data: Any,
    num_components: int,
    tol: float = 1e-5,
    max_iter: int = 500,
    random_state: int | None = None,
) -> Dict[str, np.ndarray | float]:
    """Fits a univariate Gaussian mixture model using the EM algorithm."""
    if num_components <= 0:
        raise ValueError("num_components must be positive")

    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("data must contain at least one observation")

    n_samples = x.size
    rng = np.random.default_rng(random_state)

    quantiles = np.linspace(0.0, 1.0, num_components + 2)[1:-1]
    means = np.quantile(x, quantiles)

    sample_variance = np.var(x)
    if not np.isfinite(sample_variance) or sample_variance <= 0:
        # Fall back to a small positive variance when the sample variance is degenerate.
        sample_variance = 1e-6
    variances = np.full(num_components, sample_variance, dtype=np.float64)

    weights = np.full(num_components, 1.0 / num_components, dtype=np.float64)
    responsibilities = np.full((n_samples, num_components), 1.0 / num_components, dtype=np.float64)

    log_likelihood = -np.inf

    for _ in range(max_iter):
        weighted = np.column_stack([
            weights[k] * _gaussian_pdf(x, means[k], variances[k]) for k in range(num_components)
        ])
        totals = np.clip(weighted.sum(axis=1, keepdims=True), 1e-300, None)
        responsibilities = weighted / totals

        Nk = responsibilities.sum(axis=0)
        if np.any(Nk == 0):
            Nk = np.maximum(Nk, 1e-6)

        updated_weights = Nk / n_samples
        updated_means = (responsibilities * x[:, None]).sum(axis=0) / Nk
        diffs = x[:, None] - updated_means
        updated_variances = (responsibilities * (diffs ** 2)).sum(axis=0) / Nk
        updated_variances = np.clip(updated_variances, 1e-6, None)

        weighted_updated = np.column_stack([
            updated_weights[k] * _gaussian_pdf(x, updated_means[k], updated_variances[k])
            for k in range(num_components)
        ])
        totals_updated = np.clip(weighted_updated.sum(axis=1, keepdims=True), 1e-300, None)
        new_log_likelihood = float(np.sum(np.log(totals_updated)))

        weights = updated_weights
        means = updated_means
        variances = updated_variances
        responsibilities = weighted_updated / totals_updated

        if np.isfinite(log_likelihood) and abs(new_log_likelihood - log_likelihood) < tol:
            log_likelihood = new_log_likelihood
            break

        log_likelihood = new_log_likelihood
    else:
        # Ensure log_likelihood reflects the final parameters even if convergence criterion was not met.
        log_likelihood = float(np.sum(np.log(totals_updated)))

    return {
        "weights": weights,
        "means": means,
        "variances": variances,
        "log_likelihood": log_likelihood,
        "responsibilities": responsibilities,
    }


__all__ = ["expectation_maximization_gaussian_mixture"]
