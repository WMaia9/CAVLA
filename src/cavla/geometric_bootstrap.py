"""Paired scene-cluster uncertainty for fixed fitted cross-validation models."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--group-key',choices=['scene','log_token'],default='scene')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    study=Path(json.loads((args.calibration/'manifest.json').read_text())['study'])
    config=json.loads((study/'manifest.json').read_text())['config'];data=Path(config['data'])
    with np.load(data/'dataset.npz') as z:
        oid=z['object_ids'];target=z['targets'];modes=z['query_modes'];folds=z['object_folds']
    objects=json.loads((data/'objects.json').read_text());scene=np.array([r[args.group_key] for r in objects])[oid]
    methods=config['methods'];seeds=config['seeds'];rng=np.random.default_rng(20261021)
    boot={m:np.zeros(10000) for m in methods};point={m:0. for m in methods};checks=[]
    for fold in range(3):
        scene_names=sorted({r[args.group_key] for r in objects if r['fold']==fold})
        draws=rng.integers(len(scene_names),size=(10000,len(scene_names)))
        weights=np.zeros((10000,len(scene_names)))
        np.add.at(weights,(np.arange(10000)[:,None],draws),1)
        for method in methods:
            counts=np.zeros((len(scene_names),len(seeds),4)) # FN, FP, positives, negatives
            for j,seed in enumerate(seeds):
                with np.load(args.calibration/f'fold{fold}_{method}_seed{seed}_predictions.npz') as z:
                    indices=z['indices'];prob=z['probability'].astype(float);threshold=float(z['threshold'])
                use=modes[indices]=='ood';indices=indices[use];pred=prob[use]>=threshold;y=target[indices].astype(bool)
                assert np.all(folds[oid[indices]]==fold)
                for i,name in enumerate(scene_names):
                    mask=scene[indices]==name;yi=y[mask];pi=pred[mask]
                    counts[i,j]=[(yi&~pi).sum(),(~yi&pi).sum(),yi.sum(),(~yi).sum()]
            pooled=counts.sum(0);point[method]+=float(((pooled[:,0]/pooled[:,2]+pooled[:,1]/pooled[:,3])/2).mean())/3
            sampled=(weights@counts.reshape(len(scene_names),-1)).reshape(10000,len(seeds),4)
            if np.any(sampled[:,:,2:]<=0):raise RuntimeError('Bootstrap sample with missing class')
            boot[method]+=((sampled[:,:,0]/sampled[:,:,2]+sampled[:,:,1]/sampled[:,:,3])/2).mean(1)/3
    summary=json.loads((args.calibration/'summary.json').read_text())
    for method in methods:
        assert abs(point[method]-summary[method]['ood']['balanced_error'])<1e-10
    comparisons={}
    for baseline in [m for m in methods if m!='geometry_spatial']:
        difference=boot['geometry_spatial']-boot[baseline]
        comparisons[baseline]={'point_difference':point['geometry_spatial']-point[baseline],
            'cluster_bootstrap_95_percentile':np.quantile(difference,[.025,.975]).tolist()}
    result={'point_balanced_error':point,'comparisons_geometry_spatial_minus_baseline':comparisons,
        'replicates':10000,'seed':20261021,'unit':args.group_key+', stratified by outer fold; same resample paired across methods',
        'limitations':['Conditional on these fitted models and exploratory protocol',
            'Does not account for full retraining uncertainty or prior hypothesis/method selection',
            'Cross-validation training sets overlap; this is not an independent confirmatory test']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
