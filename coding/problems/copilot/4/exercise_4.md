# Exercise 4: Regularized Logistic Regression with Preprocessing

**Difficulty:** hard
**Estimated time:** 45 minutes

## Prompt

Implement a function `train_logistic_regression_with_preprocessing(df, target_col, numeric_cols, categorical_cols, lr=0.1, l2=1.0, max_iter=2000, tol=1e-6, random_state=None)` that trains a binary logistic regression model from scratch with L2 regularization and a small tabular preprocessing pipeline.

Requirements:

1. Preprocessing
   - Numeric columns: impute missing values with the median computed from the input data, then standardize to zero mean and unit variance (guard against zero variance by using 1.0 instead).
   - Categorical columns: impute missing values with the most frequent category, then one-hot encode using categories observed in the input data (sorted ascending for determinism).
   - Concatenate the processed numeric and one-hot categorical features into a design matrix X (do not include the intercept in X; use a separate bias term).
2. Logistic regression training (from scratch, no external ML libraries)
   - Use batch gradient descent to minimize the regularized negative log-likelihood.
   - Sigmoid: σ(z) = 1 / (1 + exp(-z)); clip probabilities to the interval 1e-8 to (1 - 1e-8) for numerical stability.
   - Objective: L = -mean(y*log(p) + (1-y)*log(1-p)) + 0.5*l2*||w||^2
   - Gradients: dL/dw = X.T @ (p - y) / n + l2*w; dL/db = mean(p - y)
   - Stop when absolute change in objective < tol or when max_iter is reached.
3. Return a dictionary with keys:
   - `weights` (numpy.ndarray), `bias` (float), `feature_names` (list of strings)
   - `medians_` (dict), `means_` (dict), `stds_` (dict), `categories_` (dict mapping column to list of category strings)
   - `train_auc` (float), `n_iter` (int), `converged` (bool)
4. You may assume the target is binary with values 0/1 (or booleans). Any other truthy/falsy values should be coerced to 0/1.

## Test Case

```python
import numpy as np
import pandas as pd
from exercise_4 import train_logistic_regression_with_preprocessing

# Construct a small dataset with signal in both numeric and categorical features
rng = np.random.default_rng(123)

n0, n1 = 120, 180
x1_0 = rng.normal(-1.0, 0.5, size=n0)
x1_1 = rng.normal( 1.0, 0.5, size=n1)

x2_0 = rng.normal(0.0, 1.0, size=n0)
x2_1 = rng.normal(0.6, 1.0, size=n1)

cat_0 = rng.choice(["A", "B"], size=n0, p=[0.8, 0.2])
cat_1 = rng.choice(["A", "B"], size=n1, p=[0.3, 0.7])

# Inject some missingness
x1 = np.concatenate([x1_0, x1_1])
x2 = np.concatenate([x2_0, x2_1])
cat = np.concatenate([cat_0, cat_1]).astype(object)
y = np.concatenate([np.zeros(n0), np.ones(n1)])

x1[::25] = np.nan
cat[::40] = None

df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})

result = train_logistic_regression_with_preprocessing(
    df,
    target_col="y",
    numeric_cols=["x1", "x2"],
    categorical_cols=["cat"],
    lr=0.2,
    l2=0.5,
    max_iter=3000,
    tol=1e-7,
    random_state=7,
)

assert result["converged"] is True
assert result["n_iter"] <= 3000
assert 0.9 <= result["train_auc"] <= 1.0
assert np.isfinite(result["bias"]) and np.all(np.isfinite(result["weights"]))
print("AUC:", result["train_auc"])  # Example output: AUC: 0.96
```