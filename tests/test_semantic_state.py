import unittest

import numpy as np

from cavla.semantic_state_probe import auc, bootstrap_auc, fit_logistic, within_scene_auc
from scipy.special import expit


class SemanticStateTests(unittest.TestCase):
    def test_cluster_bootstrap_matches_literal_log_resampling_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1], bool)
        p = np.array([.2, .8, .8, .8, .2, .2])
        logs = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
        draws = np.array([[0, 0, 1], [0, 1, 2], [2, 2, 2]])
        values = bootstrap_auc(y, p, logs, draws)
        for d, value in zip(draws, values):
            ids = np.concatenate([np.flatnonzero(logs == 'abc'[j]) for j in d])
            self.assertAlmostEqual(value, auc(y[ids], p[ids]))

    def test_logistic_solver_recovers_separable_signal(self):
        x = np.arange(-20, 21, dtype=float)[:, None] / 10
        y = x[:, 0] > 0
        coefficient = fit_logistic(x, y, .01)
        predicted = expit(np.c_[np.ones(len(x)), x] @ coefficient)
        self.assertGreater(auc(y, predicted), .99)

    def test_auc_is_undefined_for_single_class_and_chance_for_ties(self):
        self.assertIsNone(auc(np.ones(3, bool), np.arange(3)))
        self.assertEqual(auc(np.array([0, 1, 0, 1], bool), np.ones(4)), .5)

    def test_scene_only_score_cannot_discriminate_agents_within_a_scene(self):
        # Scene prevalence alone creates strong global AUC despite zero
        # object-specific information in the score.
        y=np.array([0]*9+[1]+[0]+[1]*9,bool)
        scenes=np.array(['a']*10+['b']*10);score=np.r_[np.repeat(.1,10),np.repeat(.9,10)]
        self.assertGreater(auc(y,score),.8)
        result=within_scene_auc(y,score,scenes)
        self.assertEqual(result['mixed_scenes'],2)
        self.assertEqual(result['within_scene_mean_auc'],.5)


if __name__ == '__main__':
    unittest.main()
