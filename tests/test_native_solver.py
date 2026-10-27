import unittest
import numpy as np
import torch
from cavla.native_solver_probe import integrate


class NativeSolverTests(unittest.TestCase):
    def test_constant_velocity_and_initial_tensor_not_mutated(self):
        initial=torch.zeros(1,3,2)
        for method in ['euler','heun']:
            result,states,velocities=integrate(lambda x,t:torch.ones_like(x)*2,initial,10,method)
            torch.testing.assert_close(result,torch.ones_like(initial)*2)
            torch.testing.assert_close(initial,torch.zeros_like(initial))
            self.assertEqual(states.shape,(10,1,3,2));np.testing.assert_allclose(velocities,2)

    def test_heun_converges_at_second_order_on_known_flow(self):
        initial=torch.ones(1,2,2,dtype=torch.float64)
        errors=[]
        for steps in [8,16,32]:
            result,_,_=integrate(lambda x,t:x,initial,steps,'heun')
            errors.append(abs(float(result[0,0,0])-np.e))
        self.assertGreater(errors[0]/errors[1],3.5)
        self.assertGreater(errors[1]/errors[2],3.5)

    def test_rk4_converges_at_fourth_order_on_known_flow(self):
        initial=torch.ones(1,2,2,dtype=torch.float64);errors=[]
        for steps in [4,8,16]:
            result,_,_=integrate(lambda x,t:x,initial,steps,'rk4')
            errors.append(abs(float(result[0,0,0])-np.e))
        self.assertGreater(errors[0]/errors[1],14.)
        self.assertGreater(errors[1]/errors[2],14.)

    def test_cached_euler_evaluates_only_endpoint_steps(self):
        calls=[]
        def field(x,t):calls.append(float(t[0,0,0]));return torch.ones_like(x)*(1+float(t[0,0,0]))
        result,_,velocities=integrate(field,torch.zeros(1,1,1,dtype=torch.float64),8,'cached_euler')
        self.assertEqual(len(calls),4);np.testing.assert_allclose(calls,[0,.125,.75,.875])
        np.testing.assert_allclose(velocities[2:6].ravel(),1.125)
        self.assertAlmostEqual(float(result),(1+1.125*5+1.75+1.875)/8)


if __name__=='__main__':unittest.main()
