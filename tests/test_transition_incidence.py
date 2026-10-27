import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from cavla.transition_incidence_control import incidence_objective,fit_incidence
from cavla.transition_clock_probe import EDGES,path_loss


class TransitionIncidenceTests(unittest.TestCase):
    def test_objective_gradient_including_partial_followup(self):
        rng=np.random.default_rng(13);nbin=len(EDGES)-1
        x=np.c_[np.ones(9),rng.normal(size=(9,4))]
        event=np.array([.5,1.,np.nan,6.2,np.nan,3.2,np.nan,np.nan,np.nan])
        follow=np.array([.5,1.,1.2,6.2,6.4,3.2,0.,2.9,4.])
        theta=rng.normal(size=nbin+x.shape[1]);weights=np.full(9,1/9);anchor=np.zeros(nbin)
        fn=lambda t:incidence_objective(t,x,event,follow,weights,.1,anchor)
        value,grad=fn(theta);numeric=approx_derivative(lambda t:fn(t)[0],theta).ravel()
        np.testing.assert_allclose(grad,numeric,atol=1e-8,rtol=1e-6)

    def test_no_followup_contributes_no_likelihood(self):
        nbin=len(EDGES)-1;theta=np.zeros(nbin+1)
        loss,grad=incidence_objective(theta,np.ones((1,1)),np.array([np.nan]),np.array([0.]),np.ones(1),0.,np.zeros(nbin))
        self.assertEqual(loss,0.);np.testing.assert_array_equal(grad,np.zeros_like(grad))

    def test_incidence_and_hazard_likelihood_agree(self):
        from scipy.special import softmax,expit
        nbin=len(EDGES)-1;rng=np.random.default_rng(19)
        x=np.c_[np.ones(4),rng.normal(size=4)]
        theta=rng.normal(size=nbin+2);event=np.array([1.,np.nan,6.1,np.nan]);follow=np.array([1.,1.2,6.1,6.4])
        mass=expit(x@theta[nbin:])[:,None]*softmax(theta[:nbin]);surv=1-np.cumsum(mass,axis=1)
        hz=mass/np.c_[np.ones(4),surv[:,:-1]]
        value,_=incidence_objective(theta,x,event,follow,np.full(4,.25),0.,theta[:nbin])
        self.assertAlmostEqual(value,path_loss(event,follow,hz).mean(),places=10)


if __name__=='__main__':unittest.main()
