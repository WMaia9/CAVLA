"""Verify survivor pilot predictions and bootstrap paired log clusters."""
import argparse
import json
from pathlib import Path

import numpy as np

from cavla.geometric_calibration import optimal_threshold, threshold_metrics
from cavla.survivor_learning import METHODS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.study/'manifest.json').read_text())['config']
    reference=Path(config['reference_study']);rc=json.loads((reference/'manifest.json').read_text())['config']
    data=Path(rc['data']);objects=json.loads((data/'objects.json').read_text())
    with np.load(data/'dataset.npz') as z:
        oid=z['object_ids'];target=z['targets'];modes=z['query_modes'];folds=z['object_folds']
    with np.load(config['edited_features']) as z:eligible=z['eligible']
    with np.load(config['original_features']) as z:features=z['features'].reshape(len(objects),-1)
    logs=np.array([r['log_token'] for r in objects])[oid];rows=json.loads((args.study/'results.json').read_text())
    assert len(rows)==len(METHODS)*3*len(config['seeds'])*6
    max_error=0.;max_reference_error=0.;seeds=config['seeds'];rng=np.random.default_rng(20261021)
    views=['original_all','original_survivors','edited_survivors'];point={};boot={};splits=[]
    for view in views:
        point[view]={m:0. for m in METHODS};boot[view]={m:np.zeros(10000) for m in METHODS}
    for fold in range(3):
        split=json.loads((args.study/f'fold{fold}_split.json').read_text())
        groups=[{objects[i]['log_token'] for i in split[k]} for k in ['train_objects','validation_objects','test_objects']]
        for a,b in [(0,1),(0,2),(1,2)]:assert not groups[a]&groups[b]
        splits.append([len(g) for g in groups])
        with np.load(args.study/f'fold{fold}_pca.npz') as z:
            assert np.max(np.abs(z['mean']-features[split['train_objects']].mean(0)))<1e-6
        names=sorted(groups[2]);draws=rng.integers(len(names),size=(10000,len(names)))
        weights=np.zeros((10000,len(names)));np.add.at(weights,(np.arange(10000)[:,None],draws),1)
        counts={v:{m:np.zeros((len(names),len(seeds),4)) for m in METHODS} for v in views}
        for method in METHODS:
            for sj,seed in enumerate(seeds):
                name=f'fold{fold}_{method}_seed{seed}'
                with np.load(args.study/f'{name}_predictions.npz') as z:
                    i=z['indices'];po=z['original_probability'];pe=z['edited_probability'];threshold=float(z['threshold'])
                    vi=z['validation_indices'];vp=z['validation_probability']
                assert np.all(folds[oid[i]]==fold)
                assert set(vi)==set(np.flatnonzero(np.isin(oid,split['validation_objects'])&(modes=='id')))
                assert threshold==optimal_threshold(target[vi],vp)[0]
                assert np.isfinite(po).all() and np.isfinite(pe).all()
                assert np.array_equal(po[~eligible[oid[i]]],pe[~eligible[oid[i]]])
                if method=='original_repeat':
                    with np.load(reference/f'fold{fold}_geometry_spatial_seed{seed}_predictions.npz') as z:
                        assert np.array_equal(i,z['indices'])
                        reference_error=float(np.max(np.abs(po-z['probability'])))
                        assert reference_error<1e-5;max_reference_error=max(max_reference_error,reference_error)
                history=json.loads((args.study/f'{name}_validation.json').read_text());best=min(history,key=lambda r:r['balanced_error'])
                for view in views:
                    probability=pe if view=='edited_survivors' else po
                    for mode in ['id','ood']:
                        select=modes[i]==mode
                        if view!='original_all':select=select&eligible[oid[i]]
                        measured=threshold_metrics(target[i][select],probability[select],threshold)
                        row=next(r for r in rows if r['method']==method and r['seed']==seed and r['fold']==fold and r['view']==view and r['mode']==mode)
                        assert row['selected_step']==best['step']
                        for key in ['balanced_error','auc','false_negative_rate','false_positive_rate']:
                            error=abs(measured[key]-row[key]);assert error<1e-10;max_error=max(max_error,error)
                        if mode=='ood':
                            for j,log in enumerate(names):
                                use=select&(logs[i]==log);y=target[i][use].astype(bool);pred=probability[use].astype(float)>=threshold
                                counts[view][method][j,sj]=[(y&~pred).sum(),(~y&pred).sum(),y.sum(),(~y).sum()]
        for view in views:
            for method in METHODS:
                c=counts[view][method];pooled=c.sum(0)
                point[view][method]+=float(((pooled[:,0]/pooled[:,2]+pooled[:,1]/pooled[:,3])/2).mean())/3
                sample=(weights@c.reshape(len(names),-1)).reshape(10000,len(seeds),4)
                if np.any(sample[:,:,2:]<=0):raise RuntimeError('Bootstrap sample missing class')
                boot[view][method]+=((sample[:,:,0]/sample[:,:,2]+sample[:,:,1]/sample[:,:,3])/2).mean(1)/3
    summary=json.loads((args.study/'summary.json').read_text());comparisons={}
    for view in views:
        comparisons[view]={}
        for method in METHODS:
            assert abs(point[view][method]-summary[method][view]['ood']['balanced_error'])<1e-10
        for method,baseline in [(m,'original_repeat') for m in METHODS[1:]]+[(m,'norm_matched_noise') for m in ['editor_augmentation','paired_consistency']]:
            delta=boot[view][method]-boot[view][baseline]
            comparisons[view][method+'_minus_'+baseline]={'difference':point[view][method]-point[view][baseline],
                'conditional_log_bootstrap_95':np.quantile(delta,[.025,.975]).tolist()}
    context_change={}
    for method in METHODS:
        delta=boot['edited_survivors'][method]-boot['original_survivors'][method]
        context_change[method]={'difference':point['edited_survivors'][method]-point['original_survivors'][method],
            'conditional_log_bootstrap_95':np.quantile(delta,[.025,.975]).tolist()}
    result={'verified_models':3*len(METHODS)*len(seeds),'max_saved_metric_error':max_error,
        'original_repeat_vs_previous_study_maxabs_probability':max_reference_error,
        'verified':'Log-disjoint splits, train-original-only PCA mean, original ID validation indices, checkpoint choice, threshold optimality, saved metrics',
        'train_validation_test_log_counts':splits,'point_ood_balanced_error':point,'comparisons':comparisons,
        'within_method_edited_minus_original_survivors':context_change,
        'bootstrap':{'unit':'Log, paired and stratified by outer fold','replicates':10000,'seed':20261021,
            'limitations':['Fixed fitted models, no full retraining uncertainty','Exploratory reused sample; no hypothesis-selection correction','CV training sets overlap']}}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
