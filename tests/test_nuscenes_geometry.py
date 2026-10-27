import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cavla.nuscenes_geometry import iter_array, projected_box, signed_point_box_distance


class GeometryTests(unittest.TestCase):
    def test_streaming_crosses_string_and_object_boundaries(self):
        rows=[{'token':'brace } and quote " and café','value':[1,2,3]}, {'token':'x'}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'table.json';path.write_text(json.dumps(rows,ensure_ascii=False))
            self.assertEqual(list(iter_array(path,chunk_size=7)),rows)

    def test_streaming_rejects_truncated_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'table.json';path.write_text('[{"token": "unterminated')
            with self.assertRaises((ValueError,json.JSONDecodeError)):
                list(iter_array(path,chunk_size=4))

    def test_projection_known_camera(self):
        identity={'rotation':[1,0,0,0],'translation':[0,0,0]}
        sensor={**identity,'camera_intrinsic':[[100,0,100],[0,100,100],[0,0,1]]}
        annotation={'rotation':[1,0,0,0],'translation':[0,0,10],'size':[2,2,2]}
        np.testing.assert_allclose(projected_box(annotation,identity,sensor,200,200),
            [100-100/9,100-100/9,100+100/9,100+100/9])
        annotation['translation']=[0,0,.5]
        self.assertIsNone(projected_box(annotation,identity,sensor,200,200))

    def test_oriented_rectangle_distance_and_size_order(self):
        points=np.array([[3,0],[0,2],[0,0],[0,3]],float)
        center=np.zeros((4,2));sizes=np.tile([2,4,1],(4,1))
        yaw=np.array([0,0,0,np.pi/2])
        np.testing.assert_allclose(signed_point_box_distance(points,center,yaw,sizes),[1,1,-1,1],atol=1e-12)


if __name__=='__main__':unittest.main()
