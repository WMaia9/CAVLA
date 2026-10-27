import unittest
import numpy as np
from cavla.transition_clock_probe import person_period,path_loss,fit_hazard,EDGES
from cavla.transition_observation_audit import observation_window


class TransitionClockTests(unittest.TestCase):
    def test_censoring_uses_only_fully_observed_periods(self):
        rows,bins,y=person_period(np.array([np.nan,np.nan,1.01]),np.array([.49,1.01,1.01]))
        np.testing.assert_array_equal(rows,[1,1,2,2,2])
        np.testing.assert_array_equal(bins,[0,1,0,1,2])
        np.testing.assert_array_equal(y,[0,0,0,0,1])

    def test_event_on_boundary_and_path_likelihood(self):
        event=np.array([1.,np.nan]);follow=np.array([1.,1.])
        hz=np.full((2,len(EDGES)-1),.2)
        np.testing.assert_allclose(path_loss(event,follow,hz),[-np.log(.8*.2),-np.log(.8**2)])

    def test_all_horizon_periods_and_empty_prefix(self):
        event=np.array([np.nan,np.nan]);follow=np.array([6.4,0.])
        rows,bins,y=person_period(event,follow)
        self.assertEqual(len(bins),13);self.assertEqual(bins[-1],12)
        np.testing.assert_array_equal(rows,np.zeros(13,int))

    def test_empty_covariate_baseline_and_no_event_bins_finite(self):
        x=np.empty((4,0));event=np.array([1.,np.nan,2.,np.nan]);follow=np.array([1.,3.,2.,6.4]);logs=np.array(['a','a','b','b'])
        hz,record=fit_hazard(x,event,follow,logs,x,event,follow,logs,x)
        self.assertEqual(record['model'],'bin_hazard')
        self.assertTrue(np.isfinite(path_loss(event,follow,hz)).all())
        self.assertTrue(((hz>0)&(hz<1)).all())

    def test_observation_window_does_not_stop_at_motion_event(self):
        times=np.arange(20,dtype=np.int64)*500000
        follow,reason=observation_window(times,3000000)
        self.assertEqual(follow,6.4);self.assertEqual(reason,'horizon_complete')

    def test_observation_window_does_not_bridge_annotation_gap(self):
        times=np.array([0,500000,1000000,2500000,3000000],np.int64)
        follow,reason=observation_window(times,500000)
        self.assertEqual(follow,.5);self.assertEqual(reason,'annotation_gap')


if __name__=='__main__':unittest.main()
