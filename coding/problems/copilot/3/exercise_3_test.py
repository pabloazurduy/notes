import unittest

import numpy as np

from exercise_3 import expectation_maximization_gaussian_mixture


class TestExpectationMaximizationGaussianMixture(unittest.TestCase):
    def test_two_component_fit(self) -> None:
        rng = np.random.default_rng(7)
        data = np.concatenate([
            rng.normal(loc=-2.0, scale=0.6, size=200),
            rng.normal(loc=3.0, scale=1.2, size=300),
        ])

        result = expectation_maximization_gaussian_mixture(
            data,
            num_components=2,
            tol=1e-6,
            max_iter=500,
            random_state=42,
        )

        weights = np.asarray(result["weights"], dtype=np.float64)
        means = np.asarray(result["means"], dtype=np.float64)
        variances = np.asarray(result["variances"], dtype=np.float64)
        responsibilities = np.asarray(result["responsibilities"], dtype=np.float64)

        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        variances = variances[order]

        np.testing.assert_allclose(weights, np.array([0.4, 0.6]), atol=0.1)
        np.testing.assert_allclose(means, np.array([-2.0, 3.0]), atol=0.2)
        np.testing.assert_allclose(variances, np.array([0.36, 1.44]), atol=0.3)

        self.assertEqual(responsibilities.shape, (500, 2))
        self.assertTrue(np.all(responsibilities >= 0))
        np.testing.assert_allclose(responsibilities.sum(axis=1), np.ones(500), atol=1e-6)
        self.assertTrue(np.isfinite(result["log_likelihood"]))

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            expectation_maximization_gaussian_mixture([], num_components=2)

        with self.assertRaises(ValueError):
            expectation_maximization_gaussian_mixture([0.0, 1.0], num_components=0)


if __name__ == "__main__":
    unittest.main()
