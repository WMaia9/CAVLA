import unittest
import torch
from cavla.commitment_probe import euler, heun


class CommitmentProbeTest(unittest.TestCase):
    def test_freeze_reuses_velocity_and_counts_evaluations(self):
        field=lambda x,t:torch.ones_like(x)*(1+t)
        x0=torch.zeros(1,1,1,dtype=torch.float64)
        full,evals_full,_=euler(field,x0,20);frozen,evals_frozen,_=euler(field,x0,20,freeze=(4,8))
        self.assertEqual((evals_full,evals_frozen),(20,16))
        # Steps 4..7 reuse the velocity from step 3 (1+3/20) instead of 1+i/20.
        self.assertAlmostEqual(float(full-frozen),sum((i-3)/20 for i in range(4,8))/20)

    def test_kick_and_endpoint_prediction(self):
        field=lambda x,t:torch.zeros_like(x)
        x0=torch.ones(1,1,2,dtype=torch.float64);direction=torch.tensor([[[1.,0.]]],dtype=torch.float64)
        x,_,endpoints=euler(field,x0,20,kick=(5,.1),kick_direction=direction,record=True)
        torch.testing.assert_close(x,torch.tensor([[[1+.1*2**.5,1.]]],dtype=torch.float64))
        self.assertEqual(len(endpoints),20)
        final,heun_endpoints=heun(lambda x,t:x,torch.ones(1,1,1,dtype=torch.float64),64)
        self.assertAlmostEqual(float(final),2.718281828,places=3);self.assertEqual(len(heun_endpoints),64)


if __name__=='__main__':unittest.main()
