"""Weighted exact-response hierarchy for deletion-robust representations.

Numerical verification of a restricted proposition; not a novelty claim or
evidence that a VLA obeys max-plus dynamics.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from cavla.maxplus_intervention_probe import closure


def band_edges(n,order):return [(v,u) for u in range(n) for v in range(u+1,min(n,u+order+2))]


def band_graph(a,order):
    result=np.full_like(a,-np.inf)
    for v,u in band_edges(len(a),order):result[v,u]=a[v,u]
    return result


def reveal_coordinates(a,order):
    return np.array([closure(a,tuple(range(u+1,v)))[v,u] for v,u in band_edges(len(a),order)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());rng=np.random.default_rng(20261021);records=[]
    for n in [5,8,12]:
        for order in range(min(4,n-2)):
            checks=0;disagreements=0
            for seed in range(20):
                a=np.full((n,n),-np.inf)
                for v in range(n):
                    for u in range(v):a[v,u]=rng.uniform(1.,1.4)
                band=band_graph(a,order)
                for size in range(order+1):
                    for removed in itertools.combinations(range(n),size):
                        np.testing.assert_allclose(closure(a,removed),closure(band,removed),atol=1e-12,rtol=0,equal_nan=True);checks+=1
                witness=tuple(range(1,order+2));v=order+2
                assert np.isfinite(closure(a,witness)[v,0]) and np.isneginf(closure(band,witness)[v,0]);disagreements+=1
                coordinates=reveal_coordinates(a,order);np.testing.assert_allclose(coordinates,[a[v,u] for v,u in band_edges(n,order)],atol=1e-12,rtol=0)
            dimension=(order+1)*n-(order+1)*(order+2)//2;assert dimension==len(band_edges(n,order))
            records.append({'nodes':n,'deletion_order':order,'kept_independent_weights':dimension,'factual_weights':n-1,'all_tested_closures_equal':checks,
                'constructed_next_order_disagreements':disagreements,'instances':20})
    # The specially chosen response coordinates recover every independent band
    # weight directly; a finite-difference rank check supplements the algebraic proof.
    n=8;order=2;a=np.full((n,n),-np.inf)
    for v in range(n):
        for u in range(v):a[v,u]=1.2
    edges=band_edges(n,order);base=reveal_coordinates(a,order);jacobian=[]
    for v,u in edges:
        altered=a.copy();altered[v,u]+=1e-5;jacobian.append((reveal_coordinates(altered,order)-base)/1e-5)
    jacobian=np.array(jacobian).T;np.testing.assert_allclose(jacobian,np.eye(len(edges)),atol=1e-9,rtol=0)
    result={'scope':'Restricted weighted ordered complete-DAG proposition; no claim of new theorem or VLA applicability','weights_interval':[1.,1.4],
        'records':records,'coordinate_jacobian_rank':int(np.linalg.matrix_rank(jacobian)),'coordinate_dimensions':len(edges),
        'limitations':['Exact full closure query family, not one selected driving action','Continuous exact representation lower bound requires continuity and access to latent graph',
            'No noisy approximate-learning rate proved','Known graph/visual side information can resolve response-data ambiguity','No representation training or driving test']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));np.savez_compressed(args.out/'coordinate_jacobian.npz',jacobian=jacobian)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
