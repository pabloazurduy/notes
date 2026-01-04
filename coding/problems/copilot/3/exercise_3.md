# Exercise 3: Expectation-Maximization for Gaussian Mixtures

**Difficulty:** very hard
**Estimated time:** 45 minutes

## Prompt

You are given a one-dimensional dataset that is believed to come from a mixture of Gaussian distributions. Implement a function `expectation_maximization_gaussian_mixture(data, num_components, tol=1e-5, max_iter=500, random_state=None)` that fits a univariate Gaussian mixture model using the Expectation-Maximization (EM) algorithm.

The function must perform the following steps:

1. Initialize the component means with evenly spaced quantiles of the data, set all variances to the sample variance, and start with uniform mixture weights. If `random_state` is provided, use it to seed any stochastic steps you introduce.
2. Perform the E-step by computing the responsibilities for each component using the current parameters.
3. Perform the M-step by updating mixture weights, component means, and variances using the responsibilities. Guard against zero variance by clamping to a minimum of `1e-6`.
4. Compute the log-likelihood at every iteration and stop when the absolute change falls below `tol` or when `max_iter` iterations are reached. Ensure the returned log-likelihood corresponds to the final parameter update.
5. Return a dictionary containing the keys `weights`, `means`, `variances`, `log_likelihood`, and `responsibilities` (`responsibilities` should be an `n_samples x num_components` NumPy array).

Assume the input `data` can be any array-like structure convertible to a one-dimensional NumPy array.

## Test Case

```python
import numpy as np
from exercise_3 import expectation_maximization_gaussian_mixture

rng = np.random.default_rng(7)
data = np.concatenate([
    rng.normal(loc=-2.0, scale=0.6, size=200),
    rng.normal(loc=3.0, scale=1.2, size=300),
])

result = expectation_maximization_gaussian_mixture(data, num_components=2, tol=1e-6, max_iter=500, random_state=42)

# Expected behaviour (values may differ slightly due to numerical precision):
# - Two mixture weights approximately [0.4, 0.6]
# - Component means close to [-2.0, 3.0]
# - Variances near [0.36, 1.44]
# - Log-likelihood that increases monotonically during training
# - Responsibilities array shaped (500, 2)
```
