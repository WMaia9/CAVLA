import unittest

import numpy as np
import torch

from cavla.survivor_learning import norm_matched_noise
from cavla.survivor_pairs import changed_fraction, overlap_area


class SurvivorPairTests(unittest.TestCase):
    def test_roi_change_screen_uses_only_selected_pixels(self):
        support=np.zeros((10,10),bool);support[:5,:5]=True
        self.assertEqual(changed_fraction(support,[5,0,10,10]),0)
        self.assertEqual(changed_fraction(support,[0,0,10,10]),.25)
        self.assertEqual(changed_fraction(support,[-2,-2,5,5]),1)
        with self.assertRaises(ValueError):changed_fraction(support,[20,20,25,25])

    def test_touching_boxes_do_not_share_area(self):
        self.assertEqual(overlap_area([0,0,5,5],[5,0,9,5]),0)
        self.assertEqual(overlap_area([0,0,5,5],[4,4,9,9]),1)

    def test_noise_control_has_exact_per_sample_norm_and_seed(self):
        norms=torch.tensor([0.,.5,3.,10.])
        a=norm_matched_noise(norms,16,torch.Generator().manual_seed(29))
        b=norm_matched_noise(norms,16,torch.Generator().manual_seed(29))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(torch.linalg.vector_norm(a,dim=-1),norms)


if __name__=='__main__':unittest.main()
