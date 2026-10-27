import unittest
import numpy as np
from cavla.physical_envelope_audit import certificates
from cavla.innovation_scale_probe import gaussian_nll, mixture_nll, select_scale
from cavla.scale_mixture_control import fit_mixture, component_loss
from cavla.conditional_scale_mixture import gate_loss_grad

class PhysicalInformationTests(unittest.TestCase):
    def test_zero_radius_recovers_binary_predicate_including_boundary(self):
        d=np.array([[3.,2.],[3.,4.],[-1.,4.]])
        p,n=certificates(d,np.zeros(2))
        np.testing.assert_array_equal(p,d.min(-1)<=2)
        np.testing.assert_array_equal(n,~p)

    def test_certificates_hold_for_translated_point_obstacle(self):
        # A point is a degenerate box; test coupled sampled trajectories too.
        rng=np.random.default_rng(17)
        centers=rng.normal(size=(50,2,2))*5
        radius=np.array([.25,1.])
        p,n=certificates(np.linalg.norm(centers,axis=-1),radius)
        for _ in range(100):
            theta=rng.uniform(-np.pi,np.pi,(50,2))
            mag=rng.uniform(0,1,(50,2))*radius
            shift=np.stack([np.cos(theta),np.sin(theta)],-1)*mag[...,None]
            labels=np.linalg.norm(centers+shift,axis=-1).min(-1)<=2
            self.assertFalse(np.any(p&~labels));self.assertFalse(np.any(n&labels))

    def test_nll_formula_and_identical_component_mixture(self):
        energy=np.array([[0.,1.],[2.,3.]])
        variance=np.array([[1.,2.],[3.,4.]])
        expected=np.log(2*np.pi*variance)+energy/variance
        np.testing.assert_allclose(gaussian_nll(energy,variance),expected)
        np.testing.assert_allclose(mixture_nll(energy,[variance]*3),expected)

    def test_scale_readout_learns_known_heteroscedastic_signal(self):
        x=np.linspace(-2,2,40)[:,None]
        val=np.array([[-1.3],[-.4],[.4],[1.3]])
        energy=np.repeat(np.exp(2*x),4,axis=1)
        ve=np.repeat(np.exp(2*val),4,axis=1)
        pred,selected=select_scale('linear',x,energy,val,ve,[val],np.array(['a','a','b','b']))
        self.assertEqual(selected['correction'],'linear')
        self.assertLess(np.abs(np.log(pred[0]/ve)).mean(),.08)

    def test_constant_scale_retains_global_baseline(self):
        x=np.arange(20)[:,None]/10
        pred,selected=select_scale('linear',x,np.ones((20,4)),x[:2],np.ones((2,4)),[x[-2:]],np.array(['a','b']))
        self.assertEqual(selected['correction'],'global_variance')
        np.testing.assert_allclose(pred[0],1)

class MixtureControlTests(unittest.TestCase):
    def test_em_preserves_normalization_and_improves_objective(self):
        rng=np.random.default_rng(17)
        scales=np.where(rng.random(500)<.75,.1,12.)
        energy=(rng.normal(size=(500,2))**2).sum(1)*scales/2
        energy=np.repeat(energy[:,None],4,axis=1)
        variance,weights,history=fit_mixture(energy,np.array(['a']*250+['b']*250),2)
        np.testing.assert_allclose(weights.sum(0),1)
        self.assertTrue((variance>=.01).all())
        self.assertLess(np.diff(history).max(),1e-8)
        base=gaussian_nll(energy,energy.mean(0)).mean()
        self.assertLess(component_loss(energy,variance,weights).mean(),base-.2)

    def test_conditional_gate_analytic_gradient(self):
        from scipy.optimize import check_grad
        rng=np.random.default_rng(19)
        x=rng.normal(size=(8,3));density=rng.normal(size=(8,4,2))
        offset=np.log(np.full((4,2),.5));sample_weight=np.full(8,1/8)
        flat=rng.normal(size=24)*.1
        fun=lambda v:gate_loss_grad(v,x,density,offset,sample_weight,.1)[0]
        grad=lambda v:gate_loss_grad(v,x,density,offset,sample_weight,.1)[1]
        self.assertLess(check_grad(fun,grad,flat),1e-6)

if __name__=='__main__':unittest.main()
