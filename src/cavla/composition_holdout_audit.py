"""Audit frozen-threshold holdout outputs and bootstrap held-out recordings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cavla.geometric_calibration import threshold_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.study/'manifest.json').read_text());data=Path(manifest['config']['data'])
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    scenes=json.loads((data/'scenes.json').read_text());objects=json.loads((data/'objects.json').read_text())
    train_path=Path(manifest['config']['study']);train_config=json.loads((train_path/'manifest.json').read_text())['config']
    train_objects=json.loads((Path(train_config['data'])/'objects.json').read_text())
    assert not {o['log_token'] for o in objects}&{o['log_token'] for o in train_objects}
    nq=len(d['queries']);i=np.arange(len(d['case_scenes'])*nq);ci=i//nq;qi=i%nq;si=d['case_scenes'][ci]
    card=d['case_cardinality'][ci];target=d['case_targets'].reshape(-1);modes=d['query_modes'][qi]
    ambiguous=(d['oracle_lower']!=d['oracle_upper']).reshape(-1)
    masks={'original':card==0,'single':card==1,'double':card==2,'triple':card==3,'compound':card>=2,
        'compound_ambiguous':(card>=2)&ambiguous,'compound_identified':(card>=2)&~ambiguous}
    primary=masks['compound']&(modes=='ood');logs=np.array([s['log_token'] for s in scenes])[si]
    names=sorted(set(logs[primary]));methods=list(manifest['sources']);rng=np.random.default_rng(20261021)
    draws=rng.integers(len(names),size=(10000,len(names)));weights=np.zeros((10000,len(names)))
    np.add.at(weights,(np.arange(10000)[:,None],draws),1)
    scene_count=np.bincount(d['case_scenes'][d['case_cardinality']>=2],minlength=len(scenes))
    cw=1/scene_count[si[primary]];point={};boot={};weighted_point={};weighted_boot={};rows=json.loads((args.study/'results.json').read_text())
    max_error=0.;invalid_fraction={}
    for method in methods:
        counts=np.zeros((len(names),9,4));weighted_counts=np.zeros_like(counts)
        for fold in range(3):
            for seed_index,seed in enumerate([17,29,43]):
                stem=f'fold{fold}_{method}_seed{seed}';source=manifest['models'][stem]
                assert hashlib.sha256(Path(source['checkpoint']).read_bytes()).hexdigest()==source['checkpoint_sha256']
                assert hashlib.sha256(Path(source['threshold_source']).read_bytes()).hexdigest()==source['threshold_source_sha256']
                with np.load(args.study/f'{stem}_predictions.npz') as z:
                    assert np.array_equal(z['indices'],i);prob=z['probability'];threshold=float(z['threshold'])
                with np.load(source['threshold_source']) as z:assert threshold==float(z['threshold'])
                assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
                for view,mask in masks.items():
                    for mode in ['id','ood']:
                        use=mask&(modes==mode)
                        if not use.any():continue
                        measured=threshold_metrics(target[use],prob[use],threshold)
                        row=next(r for r in rows if r['method']==method and r['fold']==fold and r['seed']==seed and r['view']==view and r['mode']==mode)
                        for key in ['balanced_error','auc','false_negative_rate','false_positive_rate']:
                            if measured[key] is None:assert row[key] is None
                            else:
                                error=abs(measured[key]-row[key]);assert error<1e-10;max_error=max(max_error,error)
                yp=target[primary];pred=prob[primary].astype(float)>=threshold
                for j,name in enumerate(names):
                    use=logs[primary]==name;y=yp[use];pr=pred[use];w=cw[use];slot=fold*3+seed_index
                    counts[j,slot]=[(y&~pr).sum(),(~y&pr).sum(),y.sum(),(~y).sum()]
                    weighted_counts[j,slot]=[w[y&~pr].sum(),w[~y&pr].sum(),w[y].sum(),w[~y].sum()]
        for c,points,boots in [(counts,point,boot),(weighted_counts,weighted_point,weighted_boot)]:
            pooled=c.sum(0);points[method]=float(((pooled[:,0]/pooled[:,2]+pooled[:,1]/pooled[:,3])/2).mean())
            sampled=(weights@c.reshape(len(names),-1)).reshape(10000,9,4)
            valid=np.all(sampled[:,:,2:]>0,axis=(1,2));invalid_fraction[method]=float((~valid).mean())
            values=np.full(10000,np.nan)
            values[valid]=((sampled[valid,:,0]/sampled[valid,:,2]+sampled[valid,:,1]/sampled[valid,:,3])/2).mean(1)
            boots[method]=values
    summary=json.loads((args.study/'summary.json').read_text())
    for m in methods:assert abs(point[m]-summary[m]['compound']['ood']['balanced_error'])<1e-10
    comparisons={};weighted_comparisons={}
    for baseline in [m for m in methods if m!='geometry_joint']:
        for points,boots,output in [(point,boot,comparisons),(weighted_point,weighted_boot,weighted_comparisons)]:
            delta=boots['geometry_joint']-boots[baseline];valid=np.isfinite(delta)
            output[baseline]={'difference':points['geometry_joint']-points[baseline],
                'conditional_log_bootstrap_95':np.quantile(delta[valid],[.025,.975]).tolist() if valid.mean()>=.95 else None,
                'defined_bootstrap_fraction':float(valid.mean())}
    result={'holdout_scenes':len(scenes),'holdout_logs':len({s['log_token'] for s in scenes}),
        'compound_scenes':len(set(si[primary])),'compound_logs':len(names),'compound_ood_query_cases':int(primary.sum()),
        'max_saved_metric_error':max_error,'thresholds':'Exact originals from pre-holdout validation files, hashes verified',
        'primary_balanced_error':point,'geometry_joint_minus_baseline':comparisons,
        'scene_normalized_balanced_error':weighted_point,'scene_normalized_geometry_joint_minus_baseline':weighted_comparisons,
        'bootstrap':{'unit':'Held-out log, paired across methods, average over nine fixed predictors','replicates':10000,'seed':20261021,
            'undefined_class_draw_fraction':invalid_fraction,'limitations':['Small number of scenes/logs in compound subset',
                'Does not account for retraining uncertainty','Same underlying dataset and changed agent-subset sampling',
                'Interval reported only when at least 95% of bootstrap draws have both classes']}}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
