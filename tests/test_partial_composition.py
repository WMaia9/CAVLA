import itertools
import unittest

import numpy as np
import torch

from cavla.partial_composition_models import log_or, or_logits, joint_pair_nll, SetModel
from cavla.partial_composition_data import instance_representatives


class PartialCompositionTests(unittest.TestCase):
    def test_agent_deduplication_uses_physical_instance_and_match_quality(self):
        objects=[{'stem':s} for s in ['a','b','c','d']]
        geometry={('s',s):{'instance_token':token,'best_iou':quality} for s,token,quality in
                  [('a','agent1',.7),('b','agent1',.9),('c','agent2',.8),('d','agent2',.8)]}
        self.assertEqual(instance_representatives('s',range(4),objects,geometry),{'agent1':1,'agent2':3})

    def test_or_matches_probability_and_has_gradients_for_extreme_logits(self):
        logits=torch.tensor([[-1000.,-1001.],[1000.,-1000.]],requires_grad=True)
        keep=torch.ones_like(logits,dtype=torch.bool)
        output=or_logits(logits,keep)
        self.assertTrue(torch.isfinite(output).all())
        (-log_or(logits[:1],keep[:1])).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0].sum()),-.9)
        x=torch.tensor([[.2,-.8,1.3]],dtype=torch.float64)
        p=1-(1-x.sigmoid()).prod(-1)
        torch.testing.assert_close(or_logits(x,torch.ones_like(x,dtype=torch.bool)).sigmoid(),p)

    def test_joint_likelihood_matches_enumeration(self):
        logits=torch.tensor([[.2,-.8,1.3]],dtype=torch.float64)
        present=torch.ones((1,3),dtype=torch.bool);removed=torch.tensor([1]);p=logits.sigmoid().numpy()[0]
        for observed in [[0,0],[1,0],[1,1]]:
            probability=0.
            for bits in itertools.product([0,1],repeat=3):
                if [int(any(bits)),int(any(bits[i] for i in [0,2]))]==observed:
                    probability+=np.prod([p[i] if b else 1-p[i] for i,b in enumerate(bits)])
            loss=joint_pair_nll(logits,present,removed,torch.tensor([observed],dtype=torch.bool))
            self.assertAlmostEqual(float(loss),-np.log(probability),places=12)

    def test_redundant_observation_carries_no_gradient_for_removed_agent(self):
        x=torch.tensor([[.2,-.8,1.3]],requires_grad=True)
        joint_pair_nll(x,torch.ones_like(x,dtype=torch.bool),torch.tensor([1]),
                       torch.tensor([[True,True]])).sum().backward()
        self.assertEqual(float(x.grad[0,1]),0.)
        self.assertTrue((x.grad[0,[0,2]]<0).all())

    def test_impossible_pair_has_zero_probability_under_exact_model(self):
        value=joint_pair_nll(torch.zeros(1,3),torch.ones(1,3,dtype=torch.bool),torch.tensor([1]),
                             torch.tensor([[False,True]]))
        self.assertTrue(torch.isinf(value).all())

    def test_empty_and_deleted_sets_are_handled_and_deepsets_is_permutation_invariant(self):
        x=torch.tensor([[.2,-.8,1.3]])
        full=torch.tensor([[True,True,True]]);smaller=torch.tensor([[False,True,True]])
        self.assertLessEqual(float(or_logits(x,smaller)),float(or_logits(x,full)))
        self.assertLess(float(or_logits(x,torch.zeros_like(full))),-100)
        torch.manual_seed(4);model=SetModel(5);context=torch.randn(1,3,5);q=torch.randn(1,3)
        torch.testing.assert_close(model(context,q,smaller),model(context[:,[2,0,1]],q,smaller[:,[2,0,1]]))


if __name__=='__main__':unittest.main()
