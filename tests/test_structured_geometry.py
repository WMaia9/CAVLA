import unittest

import numpy as np
import torch

from cavla.object_features import roi_weights
from cavla.structured_geometry_learning import GeometryHead


class StructuredGeometryTests(unittest.TestCase):
    def test_roi_weights_respect_fractional_patch_overlap(self):
        np.testing.assert_allclose(roi_weights([0,0,1600,900],2,2),[.25,.25,.25,.25])
        np.testing.assert_allclose(roi_weights([0,0,400,450],2,2),[1,0,0,0])
        np.testing.assert_allclose(roi_weights([400,0,1200,450],2,2),[.5,.5,0,0])

    def test_geometric_predicate_is_monotone_in_horizon(self):
        torch.manual_seed(21)
        model=GeometryHead(10)
        # The same predicted object sits 10m ahead. A straight query at 3m/s
        # reaches it at the longer horizon; no semantic or scene assumptions.
        context=torch.zeros(2,10)
        physical=torch.tensor([[1,0,0,0,10,1.8,4.5],[1,0,0,0,10,1.8,4.5]],dtype=torch.float32)
        query=torch.tensor([[3/15,0,1/6.4],[3/15,0,4/6.4]],dtype=torch.float32)
        logits=model(context,physical,query)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertGreater(float(logits[1].detach()),float(logits[0].detach()))
        logits.sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=='__main__':unittest.main()
