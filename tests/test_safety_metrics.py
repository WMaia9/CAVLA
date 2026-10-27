import unittest
import numpy as np
from cavla.physicalai_safety_metrics import corners, gap, overlap, score


class SafetyMetricsTest(unittest.TestCase):
    def test_box_overlap_and_gap(self):
        ego=corners(np.array([0.]),np.array([0.]),np.array([0.]),4.9,2.1)
        near=corners(np.array([3.]),np.array([0.]),np.array([0.]),4.,2.)
        far=corners(np.array([10.]),np.array([0.]),np.array([0.]),4.,2.)
        self.assertTrue(overlap(ego,near)[0]);self.assertFalse(overlap(ego,far)[0])
        # Corner-to-corner distance; the 0.05 m width offset adds a tiny diagonal term.
        self.assertAlmostEqual(float(gap(ego,far)[0]),10-2-2.45,places=2)

    def test_rear_end_collision_and_clear_pass(self):
        plan=np.c_[np.arange(1,65)*1.,np.zeros(64)]
        stopped={'x':np.full(40,20.),'y':np.zeros(40),'yaw':np.zeros(40),'valid':np.ones(40,bool),'length':4.,'width':2.,'class':'automobile'}
        side={**stopped,'y':np.full(40,6.)}
        self.assertTrue(score(plan,[stopped],10.)['collision'])
        result=score(plan,[side],10.);self.assertFalse(result['collision']);self.assertAlmostEqual(result['progress_m'],40.)


if __name__=='__main__':unittest.main()
