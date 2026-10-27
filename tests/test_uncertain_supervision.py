import unittest

import numpy as np
import torch

from cavla.relations import build, weak_view
from cavla.uncertain_supervision import corrupt, mixture_loss, per_pair_loss, features
from cavla.pilot import Policy


class UncertainSupervisionTests(unittest.TestCase):
    def test_corruption_is_structured_and_audits_fixed_before_labels(self):
        _, pool, _, _ = build(12, 8)
        indices = None
        for regime in ('clean','random25','structured25'):
            observed, audit, truth = corrupt(pool,regime,12)
            self.assertEqual(int((~truth).sum()), 0 if regime=='clean' else 24)
            if regime=='structured25': self.assertTrue((pool['family'][~truth]=='approach').all())
            np.testing.assert_array_equal(truth,(observed['relation']==pool['relation']).all(axis=(1,2)))
            self.assertEqual(len(set(audit['indices'])),32)
            if indices is None: indices=audit['indices']
            else: np.testing.assert_array_equal(audit['indices'],indices)
            weak=weak_view(observed,'cpu')
            self.assertEqual(set(weak),{'images','commands','relation','anchor'})
            self.assertEqual(set(audit),{'indices','good'})

    def test_per_pair_loss_masks_each_category(self):
        effects=torch.tensor([[[0.,-.1],[0.,0.]],[[.1,0.],[0.,0.]]])
        labels=torch.tensor([[[0,-1],[0,0]],[[0,0],[0,0]]])
        losses=per_pair_loss(effects,labels)
        self.assertEqual(losses[0].item(),0)
        self.assertAlmostEqual(losses[1].item(),.0025,places=7)

    def test_trust_features_never_use_hidden_actions(self):
        _,pool,_,_=build(13,8)
        model=Policy()
        before=features(model,pool,'cpu')
        pool['targets'][:]=10000
        after=features(model,pool,'cpu')
        np.testing.assert_array_equal(before,after)

    def test_matched_support_does_not_introduce_a_novel_sign(self):
        _,pool,_,_=build(14,8)
        observed,audit,truth=corrupt(pool,'structured_matched',14)
        self.assertEqual(int((~truth).sum()),24)
        self.assertTrue((observed['relation'][~truth]==0).all())
        self.assertTrue(set(np.unique(observed['relation']))<=set(np.unique(pool['relation'])))

    def test_mixture_is_finite_and_bounds_influence(self):
        losses=torch.tensor([.1,.00005,.1,.1],requires_grad=True)
        probability=torch.tensor([0.,.8,1.,.9])
        output=mixture_loss(losses,probability)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output>=0).all())
        output.sum().backward()
        self.assertTrue((losses.grad>=0).all())
        self.assertTrue((losses.grad<=1.00001).all())
        self.assertEqual(output[0].item(),0.)
        self.assertEqual(output[2].item(),losses[2].item())
        self.assertEqual(losses.grad[0].item(),0.)
        self.assertEqual(losses.grad[2].item(),1.)
        self.assertLess(losses.grad[-1].item(),.00001)


if __name__=='__main__': unittest.main()
