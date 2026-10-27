import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from cavla.transition_ego_history import ego_features


class TransitionEgoHistoryTests(unittest.TestCase):
    def test_common_world_rigid_transform_preserves_relative_history(self):
        times=np.arange(4,dtype=np.int64)*500000
        positions=np.array([[0,0,0],[1,0,0],[2,.1,0],[3,.2,0]],float)
        rots=Rotation.from_euler('z',np.array([0.,.01,.02,.03])[:,None]);quats=rots.as_quat(scalar_first=True)
        common=Rotation.from_euler('z',.7)
        a=ego_features(times,positions,quats)
        b=ego_features(times,common.apply(positions)+[20,30,1],(common*rots).as_quat(scalar_first=True))
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_physical_timestamps_determine_velocity(self):
        times=np.array([0,500000,1000000,1500000]);pos=np.c_[times*2e-6,np.zeros((4,2))]
        q=np.tile([1,0,0,0],(4,1));f=ego_features(times,pos,q)
        np.testing.assert_allclose(f[-9:].reshape(3,3),np.tile([2,0,0],(3,1)))


if __name__=='__main__':unittest.main()
