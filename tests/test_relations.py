import unittest

import numpy as np
import torch

from cavla.controlled import world_effect
from cavla.relations import build, weak_view, relation_loss, summarize, METHODS


class RelationTests(unittest.TestCase):
    def test_training_anchors_and_disjoint_evaluation(self):
        train, pool, validation, tests = build(42, 12)
        lookup = {scene: i for i, scene in enumerate(train['ids'])}
        self.assertEqual(len(set(pool['ids'])), 96)
        for i, scene in enumerate(pool['ids']):
            np.testing.assert_array_equal(pool['images'][i, [0, 2]], train['images'][lookup[scene], [0, 2]])
            np.testing.assert_array_equal(pool['targets'][i, [0, 2]], train['targets'][lookup[scene], [0, 2]])
        used = set(train['ids'])
        for data in [validation, *tests.values()]:
            self.assertFalse(used & set(data['ids']))
            used.update(data['ids'])

    def test_labels_and_valid_null_transfer(self):
        _, pool, _, tests = build(43, 12)
        for data in [pool, *tests.values()]:
            effect = world_effect(data['targets'])
            np.testing.assert_array_equal(data['relation'], np.sign(effect))
            np.testing.assert_array_equal(data['null'], (effect == 0).all(axis=(1, 2)))
        self.assertEqual(int(pool['null'].sum()), 64)
        np.testing.assert_array_equal(pool['targets'][pool['null']][:, [1, 3]],
                                      pool['targets'][pool['null']][:, [0, 2]])
        self.assertTrue((tests['lane_swap']['relation'][:, :, 1] != 0).all())

    def test_weak_view_cannot_access_counterfactual_labels(self):
        _, pool, _, _ = build(44, 12)
        before = weak_view(pool, 'cpu')
        pool['targets'][:, [1, 3]] = 10000
        after = weak_view(pool, 'cpu')
        self.assertEqual(set(before), {'images', 'commands', 'relation', 'anchor'})
        for key in before: torch.testing.assert_close(before[key], after[key])

    def test_direction_loss_and_null_gradient(self):
        effect = torch.tensor([[[.2, .02], [0., -.02]]], requires_grad=True)
        relation = torch.tensor([[[0, 1], [0, -1]]])
        loss = relation_loss(effect, relation)
        loss.backward()
        self.assertGreater(effect.grad[0, 0, 0].item(), 0)
        self.assertEqual(effect.grad[0, 0, 1].item(), 0)
        wrong = -effect.detach()
        self.assertGreater(relation_loss(wrong, relation).item(), loss.item())
        self.assertTrue(torch.isfinite(relation_loss(effect, torch.zeros_like(relation))))

    def test_gate_requires_retention_and_both_controls(self):
        rows = []
        for seed in range(5):
            for method in METHODS:
                values = dict(primary=.5 if method == 'weak_relation' else 1.,
                              id=dict(action_mse=2. if method == 'weak_relation' else 1.))
                values.update({k: dict(action_mse=values['primary']) for k in ('severity', 'lane_swap', 'composition')})
                rows.append(dict(seed=seed, method=method, metrics=values))
        self.assertFalse(summarize(rows)['gates']['weak_relation']['pass_gate'])


if __name__ == '__main__': unittest.main()
