import unittest
import numpy as np
from cavla.transition_map_controls import distance_to_polygon,layer_distance


class TransitionMapTests(unittest.TestCase):
    def test_distance_and_hole_signs(self):
        outer=np.array([[-2,-2],[2,-2],[2,2],[-2,2]],float)
        hole=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
        p=np.array([[0,0],[1.5,0],[4,0],[2,0]],float)
        np.testing.assert_allclose(distance_to_polygon(p,outer,[hole]),[1.,-.5,2.,0.],atol=1e-10)

    def test_pruning_preserves_overlapping_polygon_minimum(self):
        a=np.array([[-2,-2],[2,-2],[2,2],[-2,2]],float)
        b=a+[1,0];p=np.array([[0,0],[1.5,0],[10,0],[100,0]],float)
        expected=np.minimum(distance_to_polygon(p,a),distance_to_polygon(p,b)).clip(-50,50)
        np.testing.assert_allclose(layer_distance(p,[(a,[]),(b,[])]),expected,atol=1e-10)
        np.testing.assert_allclose(layer_distance(p,[(b,[]),(a,[])]),expected,atol=1e-10)


if __name__=='__main__':unittest.main()
