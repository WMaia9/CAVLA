import unittest

import numpy as np

from cavla.real_pair_alignment import edit_diagnostic, unique_box


class AlignmentTests(unittest.TestCase):
    def test_same_confidence_is_ambiguous_even_when_ordinals_match(self):
        objects=[dict(id=1,label='car',confidence=.9,box_xyxy=[1,1,5,5]),
                 dict(id=2,label='car',confidence=.9,box_xyxy=[10,10,15,15])]
        self.assertIsNone(unique_box('01_car_conf0.90',dict(objects=objects),20,20))
        self.assertEqual(unique_box('02_car_conf0.90',dict(objects=objects[:1]),20,20),[1,1,5,5])

    def test_local_and_off_box_changes(self):
        original=np.zeros((64,64,3),dtype=np.uint8)
        edited=original.copy();edited[25,25]=100
        result=edit_diagnostic(original,edited,[24,24,32,32])
        self.assertEqual(result['outside_box_change_mass_fraction'],0)
        edited[0,0]=100
        result=edit_diagnostic(original,edited,[24,24,32,32])
        self.assertAlmostEqual(result['outside_box_change_mass_fraction'],.5)
        self.assertGreater(result['outside_box_pixel_fraction_above_8'],0)
        self.assertEqual(edit_diagnostic(original,original,[24,24,32,32])['mean_abs_rgb'],0)

    def test_dimension_mismatch_is_explicit(self):
        result=edit_diagnostic(np.zeros((8,8,3)),np.zeros((4,4,3)),None)
        self.assertEqual(result,dict(shape_match=False))


if __name__=='__main__':unittest.main()
