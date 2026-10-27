"""Budget, label-blindness and stochastic estimation checks."""
import unittest

import numpy as np

from cavla.acquisition import METHODS, Moments, ResponseBank, acquire


def bank(seed=19, budget=180, noise=.1, means=None):
    rng = np.random.default_rng(4)
    means = rng.normal(size=(3, 12, 4)) if means is None else means
    return ResponseBank(means, rng.normal(size=(12, 5)), rng.normal(size=(20, 5)),
                        noise, seed, budget, 8)


class BudgetAndQueries(unittest.TestCase):
    def test_every_strategy_accounts_for_all_queries(self):
        for method in METHODS:
            result = acquire(bank(), method)
            accounted = result["n_labels"]*8+result["response_queries"]+result["feature_queries"]
            self.assertEqual(result["spent"], accounted)
            self.assertLessEqual(accounted, 180)
            self.assertEqual(len(set(result["selected"])), result["n_labels"])

    def test_query_randomness_is_independent_of_request_order(self):
        a, b = bank(), bank()
        expected = a.query(2, 0)
        b.query(3, 1)
        np.testing.assert_array_equal(expected, b.query(2, 0))

    def test_query_cannot_overspend(self):
        b = bank(budget=1)
        b.query(0, 0)
        with self.assertRaises(ValueError):
            b.query(0, 0)
        self.assertEqual(b.spent, 1)

    def test_duplicate_labels_are_not_charged_twice(self):
        b = bank()
        b.label(1)
        with self.assertRaises(ValueError):
            b.label(1)
        self.assertEqual(b.spent, 8)

    def test_random_selector_never_observes_policy_outputs(self):
        result = acquire(bank(budget=40), "random")
        self.assertEqual(result["n_labels"], 5)
        self.assertEqual(result["response_queries"], 0)
        self.assertEqual(result["feature_queries"], 0)

    def test_reference_labels_are_absent_from_selector_interface(self):
        b = bank()
        self.assertFalse(any("target" in key or "reference" in key for key in vars(b)))
        self.assertIsNone(b.label(0))

    def test_adaptive_extra_budget_cap_and_last_label_reserve(self):
        result = acquire(bank(budget=180, noise=10.), "adaptive_corrected")
        self.assertLessEqual(result["extra_response_queries"], int(.25*(180-72)))
        self.assertGreaterEqual(result["n_labels"], 1)


class NoiseCorrection(unittest.TestCase):
    def test_noiseless_correction_preserves_true_between_member_variance(self):
        b = bank(noise=0)
        m = Moments.empty(3, 12, 4)
        for i in range(12):
            for k in range(3):
                for _ in range(2): m.add(i, k, b.query(i, k))
        e = m.estimates()
        np.testing.assert_allclose(e["corrected"], b._means.var(axis=0).mean(-1), atol=1e-12)

    def test_finite_sample_correction_is_unbiased_in_expectation(self):
        rng = np.random.default_rng(12)
        # 4,000 independent candidates have zero true member disagreement.
        m = Moments.empty(3, 4000, 2)
        draws = rng.normal(size=(3, 4000, 2, 2))
        m.count[:] = 2
        m.total[:] = draws.sum(axis=2)
        m.squared_total[:] = (draws**2).sum(axis=2)
        e = m.estimates()
        self.assertGreater(e["raw"].mean(), .30)
        self.assertLess(abs(e["corrected"].mean()), .025)
        self.assertTrue(np.any(e["corrected"] < 0))


if __name__ == "__main__":
    unittest.main()
