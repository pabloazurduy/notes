import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)  # stabilize exp
    return 1.0 / (1.0 + np.exp(-z))


def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Rank-based AUC with tie handling
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    # Compute ranks with average for ties
    ranks = np.empty_like(y_score_sorted, dtype=np.float64)
    i = 0
    n = y_score_sorted.size
    while i < n:
        j = i
        while j + 1 < n and y_score_sorted[j + 1] == y_score_sorted[i]:
            j += 1
        rank = 0.5 * (i + j) + 1.0  # average rank (1-indexed)
        ranks[i : j + 1] = rank
        i = j + 1

    n_pos = y_true.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined; return chance level

    sum_ranks_pos = ranks[y_true_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def train_logistic_regression_with_preprocessing(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    lr: float = 0.1,
    l2: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> Dict[str, Any]:
    # Extract target and coerce to {0,1}
    y = df[target_col].values
    y = (y.astype(float) > 0).astype(np.float64)

    # Impute and standardize numeric features
    medians: Dict[str, float] = {}
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    X_num_parts: List[np.ndarray] = []

    for col in numeric_cols:
        col_vals = df[col].to_numpy(dtype=float)
        med = np.nanmedian(col_vals)
        col_vals = np.where(np.isnan(col_vals), med, col_vals)
        mu = float(np.mean(col_vals))
        sigma = float(np.std(col_vals))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        X_num_parts.append(((col_vals - mu) / sigma)[:, None])
        medians[col] = float(med)
        means[col] = mu
        stds[col] = sigma

    X_num = np.hstack(X_num_parts) if X_num_parts else np.zeros((len(df), 0))

    # Impute and one-hot encode categorical features
    categories: Dict[str, List[str]] = {}
    X_cat_parts: List[np.ndarray] = []
    feature_names: List[str] = [*(numeric_cols)]

    for col in categorical_cols:
        col_vals = df[col].astype(object).to_numpy()
        # mode imputation
        # compute mode ignoring None/NaN
        non_null = [v for v in col_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(non_null) == 0:
            mode_val = "__missing__"
        else:
            vals, counts = np.unique(non_null, return_counts=True)
            mode_val = vals[np.argmax(counts)]
        col_vals = np.array([mode_val if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in col_vals], dtype=object)

        cats = sorted([str(c) for c in np.unique(col_vals)])
        categories[col] = cats
        one_hot = np.zeros((len(df), len(cats)), dtype=np.float64)
        for j, c in enumerate(cats):
            one_hot[:, j] = (col_vals == c).astype(np.float64)
        X_cat_parts.append(one_hot)
        feature_names.extend([f"{col}={c}" for c in cats])

    X_cat = np.hstack(X_cat_parts) if X_cat_parts else np.zeros((len(df), 0))

    X = np.hstack([X_num, X_cat]) if X_num.size or X_cat.size else np.zeros((len(df), 0))

    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)
    w = rng.normal(scale=0.01, size=n_features) if n_features > 0 else np.array([], dtype=np.float64)
    b = 0.0

    def _loss_and_grad(w: np.ndarray, b: float) -> Tuple[float, np.ndarray, float]:
        z = X @ w + b
        p = _sigmoid(z)
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        # loss
        data_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * l2 * np.dot(w, w)
        loss = data_loss + reg
        # grads
        diff = (p - y)
        grad_w = (X.T @ diff) / n_samples + l2 * w
        grad_b = float(np.mean(diff))
        return float(loss), grad_w, grad_b

    prev_loss = np.inf
    converged = False
    n_iter_done = 0

    for it in range(1, max_iter + 1):
        loss, grad_w, grad_b = _loss_and_grad(w, b)
        w -= lr * grad_w
        b -= lr * grad_b
        if np.isfinite(prev_loss) and abs(prev_loss - loss) < tol:
            converged = True
            n_iter_done = it
            break
        prev_loss = loss
    else:
        n_iter_done = max_iter

    # Final predictions and AUC
    probs = _sigmoid(X @ w + b)
    auc = _auc_score(y, probs)

    return {
        "weights": w,
        "bias": float(b),
        "feature_names": feature_names,
        "medians_": medians,
        "means_": means,
        "stds_": stds,
        "categories_": categories,
        "train_auc": float(auc),
        "n_iter": int(n_iter_done),
        "converged": bool(converged),
    }
