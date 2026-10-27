import unittest

import numpy as np

from cavla.temporal_innovation_probe import select_predict


class InnovationProbeTests(unittest.TestCase):
    def test_linear_signal_is_learned_and_generalizes(self):
        x = np.arange(20, dtype=float)[:, None] / 10
        v = np.array([[.35], [.85]])
        e = np.array([[.45], [1.25]])
        y = np.repeat(3*x + 1, 128, axis=1)
        vy = np.repeat(3*v + 1, 128, axis=1)
        predictions, selected = select_predict('linear', x, y, v, [e], np.array(['a', 'b']), vy)
        self.assertEqual(selected['correction'], 'linear')
        np.testing.assert_allclose(predictions[0], np.repeat(3*e+1, 128, axis=1), atol=.01)

    def test_zero_correction_remains_eligible_with_no_innovation(self):
        x = np.arange(20, dtype=float)[:, None]
        for family in ['linear', 'rbf']:
            predictions, selected = select_predict(family, x, np.zeros((20, 128)), x[:2], [x[-2:]],
                                                   np.array(['a', 'b']), np.zeros((2, 128)))
            self.assertEqual(selected['correction'], 'zero')
            np.testing.assert_array_equal(predictions[0], np.zeros((2, 128)))


if __name__ == '__main__':
    unittest.main()
