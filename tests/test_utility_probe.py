import unittest

import numpy as np
import torch

from cavla.utility_probe import balanced, vector_gradient


class UtilityProbeTests(unittest.TestCase):
    def test_balanced_and_independent_calibration(self):
        candidate, calibration = balanced(500, 12), balanced(520, 24)
        self.assertEqual(len(candidate['ids']), 48)
        self.assertFalse(set(candidate['ids']) & set(calibration['ids']))
        _, counts = np.unique(candidate['family'], return_counts=True)
        np.testing.assert_array_equal(counts, [12]*4)

    def test_gradient_alignment_sign_predicts_small_step(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad(): model.weight.copy_(torch.tensor([[.1, -.2]]))
        x = torch.tensor([[1., 2.], [2., 1.]])
        target = torch.tensor([[.4], [.7]])
        loss = ((model(x)-target)**2).mean()
        gradient = vector_gradient(model, loss)
        before = loss.item()
        with torch.no_grad(): model.weight.sub_(1e-4*gradient.reshape_as(model.weight))
        actual = before-((model(x)-target)**2).mean().item()
        self.assertGreater(actual, 0)
        self.assertAlmostEqual(actual, 1e-4*float(gradient @ gradient), delta=1e-6)


if __name__ == '__main__': unittest.main()
