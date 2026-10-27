"""Integration checks for factual-vs-quartet input shape and split leakage."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed in this interpreter")
class AcquisitionIntegration(unittest.TestCase):
    def test_factual_features_do_not_require_counterfactual_inputs(self):
        import torch
        from cavla.active_pilot import response_and_features
        from cavla.controlled import make_dataset
        from cavla.pilot import Policy
        data = make_dataset(8, 909)
        factual = {k: v[:, [0, 2]] if k in ["images", "commands", "targets"] else v
                   for k, v in data.items()}
        response, features, prediction = response_and_features(Policy(), factual, torch.device("cpu"))
        self.assertIsNone(response)
        self.assertEqual(features.shape, (8, 16))
        self.assertEqual(prediction.shape, (8, 2, 2))

    def test_initial_labels_and_pool_references_are_separate(self):
        from cavla.active_pilot import build_datasets, flatten
        train, val, pool, tests = build_datasets(909, 24, 12, 32)
        self.assertEqual(len(flatten(train, views=[0, 2])["targets"]), 48)
        self.assertEqual(len(flatten(pool, scenes=[0, 3])["targets"]), 8)
        groups = [set(d["ids"]) for d in [train, val, pool, *tests.values()]]
        self.assertTrue(all(not (a & b) for i, a in enumerate(groups) for b in groups[i+1:]))

    def test_inference_features_do_not_read_reference_labels(self):
        import torch
        from cavla.active_pilot import response_and_features
        from cavla.controlled import make_dataset
        from cavla.pilot import Policy
        data = make_dataset(3, 909)
        unlabeled = {k: data[k] for k in ["images", "commands"]}
        response, _, _ = response_and_features(Policy(), unlabeled, torch.device("cpu"))
        self.assertEqual(response.shape, (3, 4))

    def test_extra_repeat_controls_match_the_adaptive_cost_exactly(self):
        import numpy as np
        from cavla.acquisition import ResponseBank, acquire
        from cavla.acquisition_controls import acquire_matched
        rng = np.random.default_rng(9)
        means = rng.normal(size=(3, 12, 4))
        features, known = rng.normal(size=(12, 5)), rng.normal(size=(20, 5))
        def make_bank(): return ResponseBank(means, features, known, .5, 19, 180, 8)
        adaptive = acquire(make_bank(), "adaptive_corrected")
        for method in ["matched_uniform", "matched_random"]:
            result = acquire_matched(make_bank(), adaptive["extra_response_queries"], method)
            for key in ["spent", "n_labels", "response_queries", "extra_response_queries"]:
                self.assertEqual(result[key], adaptive[key])


if __name__ == "__main__":
    unittest.main()
