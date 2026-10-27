import unittest
import numpy as np
from cavla.conditional_policy_coupling import analytic,K,SIGMA


class ConditionalPolicyCouplingTests(unittest.TestCase):
    def test_conditional_transport_preserves_policy_and_state_marginals(self):
        for process_std in [0.,.02]:
            reference,_ = analytic('iid',process_std,30)
            for method in ['conditional_09','conditional_1']:
                records,_ = analytic(method,process_std,30)
                for a,b in zip(records,reference):
                    np.testing.assert_allclose(a['state_covariance'],b['state_covariance'],atol=1e-10,rtol=0)
                    np.testing.assert_allclose(a['conditional_action_mean_slope'],K,atol=1e-10,rtol=0)
                    self.assertAlmostEqual(a['conditional_action_variance'],SIGMA**2)

    def test_marginally_normal_autoregressive_noise_changes_conditional_policy(self):
        records,_ = analytic('ar_noise',.02,30)
        error = max(np.linalg.norm(np.array(r['conditional_action_mean_slope'])-K) for r in records)
        self.assertGreater(error,.1)

    def test_zero_memory_variance_uses_fresh_noise(self):
        a,_ = analytic('conditional_1',0.,1); b,_ = analytic('iid',0.,1)
        self.assertEqual(a,b)

    def test_observing_previous_action_removes_free_smoothing_gain(self):
        reference,_ = analytic('iid',.02,100)
        records,_ = analytic('conditional_observed_previous',.02,100)
        self.assertEqual(records,reference)


if __name__=='__main__':unittest.main()
