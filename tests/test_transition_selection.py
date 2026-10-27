import unittest
import numpy as np
from cavla.transition_tree_control import mix_hazards
from cavla.transition_family_selection import validation_selection


class TransitionSelectionTests(unittest.TestCase):
    def test_event_law_mixture_preserves_every_prefix_survival(self):
        rng=np.random.default_rng(3);a=rng.uniform(.01,.3,(8,13));b=rng.uniform(.01,.3,(8,13));weight=.3
        result=mix_hazards(a,b,weight)
        np.testing.assert_allclose(np.cumprod(1-result,axis=1),(1-weight)*np.cumprod(1-a,axis=1)+weight*np.cumprod(1-b,axis=1),atol=1e-12)

    def test_mixture_endpoints_recover_reference_and_tree(self):
        a=np.full((4,13),.03);b=np.full((4,13),.2)
        np.testing.assert_allclose(mix_hazards(a,b,0.),a,atol=1e-12)
        np.testing.assert_allclose(mix_hazards(a,b,1.),b,atol=1e-12)

    def test_selection_uses_validation_records_not_test_predictions(self):
        folds=np.repeat(np.arange(5),2);pred={'a':np.full((10,3),.1),'b':np.full((10,3),.9)}
        val={(f,m):float((m=='b')!=(f%2==0)) for f in range(5) for m in pred}
        result,records=validation_selection(pred,folds,val,['a','b'])
        changed,again=validation_selection({'a':1-pred['a'],'b':1-pred['b']},folds,val,['a','b'])
        self.assertEqual(records,again)
        for f in range(5):np.testing.assert_array_equal(result[folds==f],pred['b' if f%2==0 else 'a'][folds==f])
        self.assertFalse(np.array_equal(result,changed))


if __name__=='__main__':unittest.main()
