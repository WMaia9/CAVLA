import unittest
import numpy as np
from cavla.commitment_theory_v2 import grid_from_density
from cavla.schedule_eval import density_from_agreement


class ScheduleEvalTest(unittest.TestCase):
    def test_agreement_gain_density(self):
        d=density_from_agreement(np.linspace(0,1,4,endpoint=False),np.array([.5,.5,.9,1.]))
        np.testing.assert_allclose(d,[0,.8,.2,0])

    def test_grid_from_uniform_density_is_uniform(self):
        times=np.linspace(0,1,200,endpoint=False)
        np.testing.assert_allclose(grid_from_density(times,np.ones(200),4),[0,.25,.5,.75,1],atol=1e-9)

    def test_grid_concentrates_where_density_is_high(self):
        times=np.linspace(0,1,200,endpoint=False);density=np.where(times<.5,9.,1.)
        grid=grid_from_density(times,density,4);self.assertTrue(np.all(grid[1:4]<.5))


if __name__=='__main__':unittest.main()
