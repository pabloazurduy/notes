import unittest
import numpy as np
import pandas as pd

from exercise_4 import train_logistic_regression_with_preprocessing


class TestLogisticRegressionWithPreprocessing(unittest.TestCase):
    def test_pipeline_and_training(self) -> None:
        rng = np.random.default_rng(123)

        n0, n1 = 120, 180
        x1_0 = rng.normal(-1.0, 0.5, size=n0)
        x1_1 = rng.normal(1.0, 0.5, size=n1)

        x2_0 = rng.normal(0.0, 1.0, size=n0)
        x2_1 = rng.normal(0.6, 1.0, size=n1)

        cat_0 = rng.choice(["A", "B"], size=n0, p=[0.8, 0.2])
        cat_1 = rng.choice(["A", "B"], size=n1, p=[0.3, 0.7])

        x1 = np.concatenate([x1_0, x1_1])
        x2 = np.concatenate([x2_0, x2_1])
        cat = np.concatenate([cat_0, cat_1]).astype(object)
        y = np.concatenate([np.zeros(n0), np.ones(n1)])

        # Inject missing values
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

        self.assertTrue(result["converged"])
        self.assertLessEqual(result["n_iter"], 3000)
        self.assertGreaterEqual(result["train_auc"], 0.9)
        self.assertTrue(np.isfinite(result["bias"]))
        self.assertTrue(np.all(np.isfinite(result["weights"])) )
        self.assertEqual(len(result["feature_names"]), result["weights"].shape[0])
        # Check preprocessing artifacts exist
        self.assertIn("x1", result["medians_"])
        self.assertIn("x1", result["means_"])
        self.assertIn("x1", result["stds_"])
        self.assertIn("cat", result["categories_"])

    def test_binary_coercion(self) -> None:
        # target as booleans/ints is coerced to 0/1
        df = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [1.0, 1.0, 2.0, 3.0],
            "cat": ["A", "A", "B", "B"],
            "y": [False, True, False, True],
        })
        res = train_logistic_regression_with_preprocessing(
            df, "y", ["x1", "x2"], ["cat"], lr=0.1, l2=0.1, max_iter=50, tol=1e-9, random_state=0
        )
        self.assertIn("train_auc", res)


if __name__ == "__main__":
    unittest.main()
