import unittest
import numpy as np
from cavla.transition_history_probe import randomized_pca


class TransitionPCATests(unittest.TestCase):
    def test_queries_do_not_change_fit(self):
        rng=np.random.default_rng(19)
        train=rng.normal(size=(120,9))@rng.normal(size=(9,80))
        query=rng.normal(size=(13,80))
        a=randomized_pca(train,query,components=8)
        b=randomized_pca(train,np.r_[query,np.full((5,80),1e6)],components=8)
        np.testing.assert_allclose(a,b[:len(query)],atol=1e-10)

    def test_known_low_rank_subspace_is_preserved(self):
        rng=np.random.default_rng(23)
        train=rng.normal(size=(120,6))@rng.normal(size=(6,70))
        scores=randomized_pca(train,train,components=8)
        centered=train-train.mean(0)
        np.testing.assert_allclose(scores@scores.T,centered@centered.T,atol=1e-8)


if __name__=='__main__':unittest.main()
