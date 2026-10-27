import unittest

import numpy as np

from cavla.constraint_composition import compatible_assignments, singleton_signature, composition_bounds


class ConstraintCompositionTests(unittest.TestCase):
    def test_two_redundant_agents_are_identified_by_both_deletions(self):
        np.testing.assert_array_equal(compatible_assignments([1,1]),[[1,1]])

    def test_redundancy_with_three_agents_has_four_equivalent_assignments(self):
        compatible=compatible_assignments([1,1,0])
        self.assertEqual(compatible.shape,(4,3))
        self.assertTrue((compatible.sum(1)>=2).all())
        self.assertTrue((compatible.min(0)!=compatible.max(0)).all())
        self.assertEqual(singleton_signature([1,1,0]),singleton_signature([0,1,1]))

    def test_unique_blocker_and_no_blocker_are_identified(self):
        for bits in [[0,1,0,0],[0,0,0]]:
            np.testing.assert_array_equal(compatible_assignments(bits),[bits])

    def test_composition_bound_exposes_single_deletion_ambiguity(self):
        self.assertEqual(composition_bounds(True,[True,True,True],[0,1]),
                         {'lower':0,'upper':1,'compatible_assignments':4})
        self.assertEqual(composition_bounds(True,[True,True,True],[])['lower'],1)
        self.assertEqual(composition_bounds(True,[True,True,True],[0,1,2])['upper'],0)

    def test_missing_observations_and_inconsistent_outcomes_are_explicit(self):
        self.assertEqual(composition_bounds(None,[None,None],[0])['compatible_assignments'],4)
        with self.assertRaises(ValueError):composition_bounds(False,[True,None],[])
        with self.assertRaises(ValueError):composition_bounds(True,[True,True],[2])


if __name__=='__main__':unittest.main()
