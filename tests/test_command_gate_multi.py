import unittest
import numpy as np
from cavla.native_command_gate_multi import auroc, episode_bootstrap, mode


class CommandGateMultiTest(unittest.TestCase):
    def test_mode(self):
        self.assertEqual(mode({'throttle':.6,'brake':0.}),'throttle')
        self.assertEqual(mode({'throttle':0.,'brake':.2}),'brake')
        self.assertEqual(mode({'throttle':0.,'brake':0.}),'coast')

    def test_auroc(self):
        self.assertEqual(auroc([0,1,2,3],[0,0,1,1]),1.)
        self.assertEqual(auroc([3,2,1,0],[0,0,1,1]),0.)
        self.assertEqual(auroc([1,1],[0,1]),.5)
        self.assertIsNone(auroc([1,2],[1,1]))

    def test_episode_bootstrap_constant(self):
        lo,hi=episode_bootstrap([1.,1.,1.,1.],['a','a','b','b'],np.random.default_rng(0),reps=50)
        self.assertEqual((lo,hi),(1.,1.))


if __name__=='__main__':unittest.main()
