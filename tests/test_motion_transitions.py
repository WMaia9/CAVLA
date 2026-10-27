import unittest
import numpy as np
from cavla.motion_transition_inventory import first_landmark,future_event,history_features

class MotionTransitionTests(unittest.TestCase):
    def test_first_landmark_is_prospective(self):
        times=np.arange(20,dtype=np.int64)*500000
        positions=np.zeros((20,3));positions[8:,0]=np.arange(12)*2.
        self.assertEqual(first_landmark(times,positions),6)
        self.assertEqual(first_landmark(times[:7],positions[:7]),6)

    def test_invalid_past_gap_is_not_bridged(self):
        times=np.array([0,500000,1000000,2000000,2500000,3000000,3500000])
        self.assertIsNone(first_landmark(times,np.zeros((7,3))))

    def test_requires_two_fast_future_intervals_and_displacement(self):
        times=np.arange(12,dtype=np.int64)*500000
        pos=np.zeros((12,3));pos[7:,0]=np.array([.1,1.,3.,4.,5.])
        event,follow=future_event(times,pos,6)
        self.assertEqual(event,1.5)
        self.assertEqual(follow,1.5)

    def test_short_track_remains_censored_without_event(self):
        times=np.arange(9,dtype=np.int64)*500000
        event,follow=future_event(times,np.zeros((9,3)),6)
        self.assertIsNone(event);self.assertEqual(follow,1.)

    def test_translation_does_not_change_history_features(self):
        times=np.arange(7,dtype=np.int64)*500000
        p=np.c_[np.linspace(0,.1,7),np.linspace(0,.2,7),np.zeros(7)]
        a=history_features(times,p,.3);b=history_features(times,p+[100,200,0],.3)
        for x,y in zip(a[:3],b[:3]):np.testing.assert_allclose(x,y,atol=1e-10)

if __name__=='__main__':unittest.main()
