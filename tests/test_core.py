"""Checks that matter for causal claims and experimental leakage."""
import unittest

import numpy as np

from cavla.audit_existing import matched_box, trajectory_response
from cavla.controlled import State, make_dataset, mixed_effect, oracle, world_effect
from cavla.statistics import cluster_bootstrap, grouped_folds, ridge_predict
from cavla.distribution_effects import paired_mean_effect_squared, paired_rbf_mmd_squared


class ControlledEffects(unittest.TestCase):
    def test_relevance_changes_with_command(self):
        far, close = State(-1, .8, .1, .5), State(-1, .6, .1, .5)
        np.testing.assert_allclose(oracle(close, 0)-oracle(far, 0), [0, -.2], atol=1e-6)
        np.testing.assert_array_equal(oracle(close, 1), oracle(far, 1))

    def test_null_changes_pixels_but_not_oracle(self):
        d = make_dataset(64, 12)
        self.assertTrue(d["null"].any())
        np.testing.assert_array_equal(world_effect(d["targets"])[d["null"]], 0)
        self.assertTrue(np.all(np.abs(d["images"][d["null"], 1]-d["images"][d["null"], 0]).sum((1, 2, 3)) > 0))

    def test_language_world_interaction_has_correct_sign(self):
        d = make_dataset(64, 12)
        effect = mixed_effect(d["targets"])[~d["null"], 1]
        lane = d["states"][~d["null"], 0]
        self.assertTrue(np.all(effect*lane < 0))

    def test_background_shift_preserves_targets(self):
        a, b = make_dataset(30, 12), make_dataset(30, 12, "background")
        np.testing.assert_array_equal(a["targets"], b["targets"])
        self.assertFalse(np.array_equal(a["images"], b["images"]))

    def test_auxiliary_head_can_be_perfect_with_wrong_policy(self):
        # z=(constant action component, effect component): auxiliary readout
        # accesses the latter while the policy ignores it. Perfect readout
        # therefore does not imply a faithful intervention response.
        effects = np.array([-.3, .1, .6])
        z0, z1 = np.zeros((3, 2)), np.column_stack([np.zeros(3), effects])
        aux = (z1-z0)[:, 1]
        policy_effect = (z1-z0)[:, 0]
        np.testing.assert_array_equal(aux, effects)
        self.assertGreater(np.mean((policy_effect-effects)**2), 0)


class AuditAndStatistics(unittest.TestCase):
    def test_ade_change_can_vanish_despite_large_response(self):
        original = np.array([[[1., 0., 0.]]])
        edited = np.array([[[-1., 0., 0.]]])
        paired, mean = trajectory_response(original, edited)
        self.assertEqual(paired, 2.)
        self.assertEqual(mean, 2.)
        self.assertEqual(abs(np.linalg.norm(original)-np.linalg.norm(edited)), 0.)

    def test_trajectory_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            trajectory_response(np.zeros((1, 2, 3)), np.zeros((2, 2, 3)))

    def test_no_scene_leakage(self):
        groups = np.repeat(np.arange(20), 3)
        held_out = []
        for train, test in grouped_folds(groups):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            held_out.extend(test.tolist())
        self.assertEqual(sorted(held_out), list(range(len(groups))))

    def test_cluster_weighting_is_by_scene(self):
        result = cluster_bootstrap([0., 0., 0., 10.], [0, 0, 0, 1])
        self.assertEqual(result["mean"], 5.)
        self.assertEqual(result["n_independent_scenes"], 2)

    def test_ridge_handles_missing_features_without_test_fitting(self):
        train = np.array([[0., np.nan], [1., np.nan], [2., np.nan]])
        pred = ridge_predict(train, np.array([0., 1., 2.]), [[3., 1e9]], alpha=1e-6)
        np.testing.assert_allclose(pred, [3.], atol=1e-5)

    def test_metadata_join_rejects_ambiguous_candidates(self):
        obj = {"label": "car", "confidence": .9, "box_xyxy": [1, 2, 3, 4]}
        self.assertIsNone(matched_box("02_car_conf0.90", {"objects": [obj, obj]}, 10, 10))
        self.assertIsNotNone(matched_box("02_car_conf0.90", {"objects": [obj]}, 10, 10))


class StochasticEffects(unittest.TestCase):
    def test_constant_shift_is_recovered_exactly(self):
        a = np.zeros((5, 3))
        result = paired_mean_effect_squared(a, a+2)
        self.assertEqual(result["mean_effect_squared_unbiased"], 4.)

    def test_negative_estimates_are_preserved(self):
        result = paired_mean_effect_squared(np.zeros((2, 1)), np.array([[-1.], [1.]]))
        self.assertEqual(result["mean_effect_squared_unbiased"], -1.)

    def test_mean_estimator_is_unbiased_under_independent_seed_pairs(self):
        rng = np.random.default_rng(23)
        u = rng.normal(size=(4000, 8, 1))
        # F0(U)=U and F1(U)=-U have identical normal marginals, but
        # paired squared displacement has expectation 4. Mean shift is 0.
        delta = -2*u
        estimates = ((delta.sum(1)**2-(delta**2).sum(1))/(8*7)).ravel()
        self.assertLess(abs(estimates.mean()), .06)
        self.assertGreater((delta**2).mean(), 3.8)

    def test_mmd_is_zero_for_identical_samples_with_same_pairing(self):
        a = np.random.default_rng(1).normal(size=(5, 2))
        self.assertEqual(paired_rbf_mmd_squared(a, a, 1.), 0.)

    def test_mmd_detects_distribution_shift_even_if_mean_is_equal(self):
        rng = np.random.default_rng(3)
        a = rng.normal(scale=.01, size=(100, 1))
        b = rng.choice([-2., 2.], size=(100, 1))
        self.assertGreater(paired_rbf_mmd_squared(a, b, .5), .8)


if __name__ == "__main__":
    unittest.main()
