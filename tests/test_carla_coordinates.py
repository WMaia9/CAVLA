import unittest
import numpy as np
from cavla.carla_native_contexts import local_right_handed_history


class CarlaCoordinatesTests(unittest.TestCase):
    def test_forward_history_stays_forward_at_rotated_world_heading(self):
        for yaw in [0.,.7,1.8,-2.]:
            c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            matrices=np.repeat(np.eye(4)[None],16,axis=0);matrices[:,:3,:3]=rot
            matrices[:,:3,3]=np.arange(16)[:,None]*rot[:,0][None,:]
            xyz,local_rot=local_right_handed_history(matrices)
            np.testing.assert_allclose(xyz[:,0],np.arange(-15,1),atol=1e-5)
            np.testing.assert_allclose(xyz[:,1:],0,atol=1e-5)
            np.testing.assert_allclose(local_rot,np.repeat(np.eye(3)[None],16,axis=0),atol=1e-6)

    def test_carla_right_turn_becomes_negative_right_handed_yaw(self):
        yaw=.2;c,s=np.cos(yaw),np.sin(yaw)
        matrices=np.repeat(np.eye(4)[None],2,axis=0)
        matrices[0,:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        _,rot=local_right_handed_history(matrices)
        self.assertAlmostEqual(float(np.arctan2(rot[0,1,0],rot[0,0,0])),-yaw,places=6)


if __name__=='__main__':unittest.main()
