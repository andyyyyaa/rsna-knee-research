import sys, unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rsna_research.metrics import per_class_auc, draw_counts, bootstrap_auc, percentile_rank, operating_point
from rsna_research.fusion import fusion

class TestMetrics(unittest.TestCase):
    def test_auc_ties_and_reverse(self):
        y=np.tile([0,0,1,1],(12,1)).T
        tied=np.ones_like(y,dtype=float)
        np.testing.assert_allclose(per_class_auc(y,tied),.5)
        np.testing.assert_allclose(per_class_auc(y,y),1)
        np.testing.assert_allclose(per_class_auc(y,1-y),0)

    def test_weighted_bootstrap_matches_explicit_resampling(self):
        rng=np.random.default_rng(412)
        y=rng.integers(0,2,(30,12)); p=rng.integers(0,5,(30,12))/4
        counts,_=draw_counts(y,25,17); actual=bootstrap_auc(y,p,counts)
        for c,got in zip(counts,actual):
            ix=np.repeat(np.arange(len(y)),c)
            np.testing.assert_allclose(got,roc_auc_score(y[ix],p[ix],average=None),atol=1e-12)

    def test_one_class_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): per_class_auc(np.zeros((10,12)),np.ones((10,12)))
        y=np.tile([0,1],(12,1)).T
        with self.assertRaises(ValueError): per_class_auc(y,np.full((2,12),np.nan))

    def test_rank_ties_are_average(self):
        x=np.tile([0,0,1,1],(12,1)).T
        np.testing.assert_allclose(percentile_rank(x)[:,0],[.375,.375,.875,.875])

    def test_fusion_permutation_equivariance(self):
        rng=np.random.default_rng(66)
        b={k:rng.random((58,12)) for k in ['384','224','r6','reader','extra']}
        ix=rng.permutation(58)
        np.testing.assert_array_equal(fusion({k:v[ix] for k,v in b.items()},extra='extra'),fusion(b,extra='extra')[ix])

    def test_operating_point_has_correct_denominators(self):
        y=np.tile([0,0,1,1],(12,1)).T
        op=operating_point(y,y)
        np.testing.assert_array_equal(op['tp'],2)
        np.testing.assert_array_equal(op['fn'],0)
        np.testing.assert_allclose(op['recall'],1)

if __name__=='__main__':unittest.main()
