import unittest

import numpy as np

from cavla.temporal_necessity_audit import past_state


class PastOnlyTests(unittest.TestCase):
    def test_future_annotation_cannot_change_velocity_or_position(self):
        samples = {str(i): {'timestamp': i * 500000} for i in range(4)}
        track = [{'sample_token': str(i), 'translation': [i, 0, 0],
                  'rotation': [1, 0, 0, 0], 'size': [2, 4, 2]} for i in range(4)]
        ego = {'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0]}
        center, velocity, _, _, provenance = past_state(track, samples, 750000, ego)
        np.testing.assert_allclose(center, [1.5, 0])
        np.testing.assert_allclose(velocity, [2, 0])
        track[-2]['translation'] = [100000, 100000, 0]
        changed = past_state(track, samples, 750000, ego)
        np.testing.assert_allclose(changed[0], center)
        np.testing.assert_allclose(changed[1], velocity)
        self.assertLessEqual(provenance['latest_annotation_timestamp'], provenance['input_timestamp'])

    def test_stale_history_and_only_one_past_state_are_rejected(self):
        samples = {'a': {'timestamp': 0}, 'b': {'timestamp': 500000}}
        track = [{'sample_token': s, 'translation': [0, 0, 0],
                  'rotation': [1, 0, 0, 0], 'size': [2, 4, 2]} for s in samples]
        ego = {'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0]}
        self.assertIsNone(past_state(track, samples, 100000, ego))
        self.assertIsNone(past_state(track, samples, 1500000, ego))


if __name__ == '__main__':
    unittest.main()
