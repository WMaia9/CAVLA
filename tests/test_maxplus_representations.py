import itertools
import unittest
import numpy as np
from cavla.maxplus_intervention_probe import closure, hidden_edge_pair, same, separating_cut
from cavla.maxplus_capacity_probe import band_graph, band_edges, reveal_coordinates
from cavla.maxplus_quantization_probe import encode, path_response_error
from cavla.maxplus_rate_probe import adaptive_encode


class MaxplusRepresentationsTests(unittest.TestCase):
    def test_closure_uses_longer_path_and_removes_incident_edges(self):
        a = np.full((3, 3), -np.inf)
        a[1, 0] = 1.; a[2, 1] = 1.; a[2, 0] = 1.4
        self.assertEqual(closure(a)[2, 0], 2.)
        self.assertEqual(closure(a, [1])[2, 0], 1.4)
        self.assertTrue(np.isnan(closure(a, [1])[1]).all())

    def test_single_deletions_cannot_reveal_redundant_constraint(self):
        a, b = hidden_edge_pair(3)
        for size in range(3):
            for removed in itertools.combinations(range(5), size):
                self.assertTrue(same(closure(a, removed), closure(b, removed)))
        self.assertFalse(same(closure(a, [1, 2, 3]), closure(b, [1, 2, 3])))
        cut, _ = separating_cut(a, b)
        self.assertEqual(set(cut), {1, 2, 3})

    def test_band_coordinates_and_next_order_witness(self):
        rng = np.random.default_rng(42); n = 7; k = 2
        a = np.full((n, n), -np.inf); ix = np.tril_indices(n, -1)
        a[ix] = rng.uniform(1, 1.4, len(ix[0])); b = band_graph(a, k)
        for size in range(k + 1):
            for removed in itertools.combinations(range(n), size):
                np.testing.assert_allclose(closure(a, removed), closure(b, removed))
        np.testing.assert_allclose(reveal_coordinates(a, k), [a[v, u] for v, u in band_edges(n, k)])
        self.assertFalse(same(closure(a, [1, 2, 3]), closure(b, [1, 2, 3])))

    def test_coding_bound_and_fast_error_match_full_closure(self):
        rng = np.random.default_rng(10); n = 8; k = 3; delta = .01
        a = np.full((n, n), -np.inf); ix = np.tril_indices(n, -1)
        a[ix] = rng.uniform(1, 1.4, len(ix[0]))
        for method in ['independent', 'chain_feedback', 'potential_feedback']:
            b = encode(a, k, delta, method)
            for size in range(k + 1):
                for removed in itertools.combinations(range(n), size):
                    c = closure(a, removed); d = closure(b, removed); valid = np.isfinite(c)
                    exact = np.max(np.abs(c[valid] - d[valid]))
                    self.assertAlmostEqual(path_response_error(a, b, removed), exact)
                    if method == 'potential_feedback':
                        self.assertLessEqual(exact, delta * (1 + size / 2) + 1e-10)

    def test_coding_rejects_unproved_weight_or_margin_domain(self):
        a = np.full((4, 4), -np.inf); a[np.tril_indices(4, -1)] = 1.2
        with self.assertRaises(ValueError): encode(a, 1, .2, 'potential_feedback')
        a[2, 0] = 3.
        with self.assertRaises(ValueError): encode(a, 1, .01, 'potential_feedback')

    def test_variable_precision_and_classical_control_preserve_responses(self):
        rng = np.random.default_rng(123); n = 8; k = 4; epsilon = .02
        a = np.full((n,n),-np.inf); ix = np.tril_indices(n,-1)
        a[ix] = rng.uniform(1,1.4,len(ix[0]))
        for method in ['potential','chain']:
            b,bits = adaptive_encode(a,k,epsilon,method)
            self.assertGreater(bits,0)
            for size in range(k+1):
                for removed in itertools.combinations(range(n),size):
                    c = closure(a,removed); d = closure(b,removed); valid = np.isfinite(c)
                    exact = np.max(np.abs(c[valid]-d[valid]))
                    self.assertLessEqual(exact,epsilon+1e-10)
                    self.assertAlmostEqual(exact,path_response_error(a,b,removed))


if __name__ == '__main__': unittest.main()
