"""Verify composition predictions, grouped splits and paired log uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cavla.geometric_calibration import optimal_threshold, threshold_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--metadata-study',type=Path,required=True)
    p.add_argument('--supervised-reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.study/'manifest.json').read_text())['config'];data=Path(config['data'])
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((data/'objects.json').read_text());scenes=json.loads((data/'scenes.json').read_text())
    data_manifest=json.loads((data/'manifest.json').read_text())
    dedup_geometry=data_manifest.get('deduplication',{}).get('geometry')
    if dedup_geometry:
        geometry={(r['scene'],r['stem']):r for r in json.loads((Path(dedup_geometry)/'records.json').read_text())}
        for members in d['scene_object_ids']:
            tokens=[geometry[objects[i]['scene'],objects[i]['stem']]['instance_token'] for i in members if i>=0]
            assert len(tokens)==len(set(tokens))
    nq=len(d['queries']);indices=np.arange(len(d['case_scenes'])*nq);ci=indices//nq;qi=indices%nq
    si=d['case_scenes'][ci];card=d['case_cardinality'][ci];target=d['case_targets'].reshape(-1)
    logs=np.array([s['log_token'] for s in scenes])[si];query_modes=d['query_modes'][qi]
    assert np.all(d['oracle_lower']<=d['case_targets']) and np.all(d['case_targets']<=d['oracle_upper'])
    for sj,scene in enumerate(scenes):
        original=d['object_targets'][sj].any(0);without=d['object_targets'][sj].copy()
        without[d['removed_index'][sj]]=False
        assert np.array_equal(d['weak_labels'][sj],np.stack([original,without.any(0)],-1))
        cases=np.flatnonzero(d['case_scenes']==sj)
        assert np.array_equal(d['case_targets'][cases],d['case_keep'][cases].astype(int)@d['object_targets'][sj].astype(int)>0)
        assert not np.any((d['case_cardinality'][cases]>=2)&~d['case_keep'][cases].any(-1))
    sources={m:(args.study,m) for m in config['methods']}
    sources['geometry_joint_metadata']=(args.metadata_study,'geometry_joint')
    for m in ['supervised_or','supervised_max']:sources[m]=(args.supervised_reference,m)
    rows_by_path={str(path):json.loads((path/'results.json').read_text()) for path,_ in sources.values()}
    views=['original','single','double','triple','compound','compound_ambiguous','compound_identified']
    ambiguous=(d['oracle_lower']!=d['oracle_upper']).reshape(-1)
    masks={'original':card==0,'single':card==1,'double':card==2,'triple':card==3,'compound':card>=2,
        'compound_ambiguous':(card>=2)&ambiguous,'compound_identified':(card>=2)&~ambiguous}
    with np.load(config['features']) as z:features=z['features'].reshape(len(objects),-1)
    max_error=0.;files={};point={m:0. for m in sources};weighted_point={m:0. for m in sources}
    bootstrap={m:np.zeros(10000) for m in sources};weighted_bootstrap={m:np.zeros(10000) for m in sources}
    rng=np.random.default_rng(20261021);monotonic=[];scene_count=np.bincount(d['case_scenes'][d['case_cardinality']>=2],minlength=len(scenes))
    splits=[]
    for fold in range(3):
        split=json.loads((args.study/f'fold{fold}_split.json').read_text())
        groups=[{objects[i]['log_token'] for i in split[k]} for k in ['train_objects','validation_objects','test_objects']]
        for a,b in [(0,1),(0,2),(1,2)]:assert not groups[a]&groups[b]
        splits.append([len(g) for g in groups]);names=sorted(groups[2])
        draws=rng.integers(len(names),size=(10000,len(names)));weights=np.zeros((10000,len(names)))
        np.add.at(weights,(np.arange(10000)[:,None],draws),1)
        for path in [args.study,args.metadata_study]:
            assert json.loads((path/f'fold{fold}_split.json').read_text())==split
            c=json.loads((path/'manifest.json').read_text())['config']
            with np.load(path/f'fold{fold}_pca.npz') as z:
                if not c.get('metadata_only',False):assert np.max(np.abs(z['mean']-features[split['train_objects']].mean(0)))<1e-6
        for alias,(path,method) in sources.items():
            counts=np.zeros((len(names),3,4));weighted_counts=np.zeros_like(counts)
            for seed_index,seed in enumerate([17,29,43]):
                file=path/f'fold{fold}_{method}_seed{seed}_predictions.npz'
                files[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
                with np.load(file) as z:
                    i=z['indices'];prob=z['probability'];threshold=float(z['threshold'])
                    vi=z['validation_group_ids'];vp=z['validation_probability']
                assert np.array_equal(i,np.flatnonzero(d['scene_folds'][si]==fold))
                assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
                expected=np.flatnonzero(np.isin(np.repeat(np.array([s['scene'] for s in scenes]),nq),
                    [objects[j]['scene'] for j in split['validation_objects']])&np.tile(d['query_modes']=='id',len(scenes)))
                assert np.array_equal(vi,expected)
                vy=d['weak_labels'][vi//nq,vi%nq].reshape(-1)
                assert threshold==optimal_threshold(vy,vp.reshape(-1))[0]
                source_rows=rows_by_path[str(path)]
                selected_step=None
                if alias not in ['supervised_or','supervised_max']:
                    history=json.loads((path/f'fold{fold}_{method}_seed{seed}_validation.json').read_text())
                    selected_step=min(history,key=lambda r:r['balanced_error'])['step']
                for view in views:
                    for mode in ['id','ood']:
                        choose=masks[view][i]&(query_modes[i]==mode)
                        if not choose.any():continue
                        measured=threshold_metrics(target[i][choose],prob[choose],threshold)
                        row=next(r for r in source_rows if r['method']==method and r['fold']==fold and r['seed']==seed and r['view']==view and r['mode']==mode)
                        if selected_step is not None:assert row['selected_step']==selected_step
                        for key in ['auc','balanced_error','false_negative_rate','false_positive_rate']:
                            if measured[key] is None:assert row[key] is None
                            else:
                                error=abs(measured[key]-row[key]);assert error<1e-10;max_error=max(max_error,error)
                choose=masks['compound'][i]&(query_modes[i]=='ood');ix=i[choose];pred=prob[choose].astype(float)>=threshold
                y=target[ix];cw=1/scene_count[si[ix]]
                for j,name in enumerate(names):
                    use=logs[ix]==name;yi=y[use];pi=pred[use];w=cw[use]
                    counts[j,seed_index]=[(yi&~pi).sum(),(~yi&pi).sum(),yi.sum(),(~yi).sum()]
                    weighted_counts[j,seed_index]=[w[yi&~pi].sum(),w[~yi&pi].sum(),w[yi].sum(),w[~yi].sum()]
                matrix=np.full(len(indices),np.nan);matrix[i]=prob;violations=[]
                for sj in np.flatnonzero(d['scene_folds']==fold):
                    cases=np.flatnonzero(d['case_scenes']==sj);original_case=cases[d['case_cardinality'][cases]==0][0]
                    singleton=cases[d['case_cardinality'][cases]==1]
                    for case in cases[d['case_cardinality'][cases]>=2]:
                        # Any larger exclusion must have no greater OR response.
                        removed_here=~d['case_keep'][case]&(d['scene_object_ids'][sj]>=0)
                        supersets=[c for c in singleton if np.all(d['case_keep'][case]<=d['case_keep'][c])]
                        assert len(supersets)==int(removed_here.sum())
                        bound=np.min(matrix[(np.array([original_case]+supersets)[:,None]*nq)+np.arange(nq)],axis=0)
                        violations.extend((matrix[case*nq+np.arange(nq)]>bound+1e-6)[d['query_modes']=='ood'])
                monotonic.append({'method':alias,'fold':fold,'seed':seed,'violation_fraction':float(np.mean(violations))})
            for c,point_store,boot_store in [(counts,point,bootstrap),(weighted_counts,weighted_point,weighted_bootstrap)]:
                pooled=c.sum(0);point_store[alias]+=float(((pooled[:,0]/pooled[:,2]+pooled[:,1]/pooled[:,3])/2).mean())/3
                sample=(weights@c.reshape(len(names),-1)).reshape(10000,3,4)
                if np.any(sample[:,:,2:]<=0):raise RuntimeError('Bootstrap missing a class')
                boot_store[alias]+=((sample[:,:,0]/sample[:,:,2]+sample[:,:,1]/sample[:,:,3])/2).mean(1)/3
    comparisons={};weighted_comparisons={}
    for alias,(path,method) in sources.items():
        summary=json.loads((path/'summary.json').read_text())
        assert abs(point[alias]-summary[method]['compound']['ood']['balanced_error'])<1e-10
    for baseline in [m for m in sources if m!='geometry_joint']:
        for store,points,boots in [(comparisons,point,bootstrap),(weighted_comparisons,weighted_point,weighted_bootstrap)]:
            delta=boots['geometry_joint']-boots[baseline]
            store[baseline]={'difference':points['geometry_joint']-points[baseline],
                'conditional_log_bootstrap_95':np.quantile(delta,[.025,.975]).tolist()}
    meta_config=json.loads((args.metadata_study/'manifest.json').read_text())['config']
    result={'new_trained_models':3*len(config['seeds'])*(len(config['methods'])+len(meta_config['methods'])),
        'reused_supervised_models':9,'max_saved_metric_error':max_error,
        'verified':'Algebraic labels, privileged bounds, train-only PCA, log-disjoint splits, weak validation IDs, checkpoint and threshold choices, saved metrics',
        'train_validation_test_logs':splits,'primary_compound_ood_balanced_error':point,
        'geometry_joint_minus_baseline':comparisons,'scene_normalized_compound_ood_balanced_error':weighted_point,
        'scene_normalized_geometry_joint_minus_baseline':weighted_comparisons,
        'monotonic_violation_fraction':{m:float(np.mean([r['violation_fraction'] for r in monotonic if r['method']==m])) for m in sources},
        'bootstrap':{'unit':'Log, paired and stratified by outer fold','replicates':10000,'seed':20261021,
            'limitations':['Fixed fitted models, no retraining uncertainty','Exploratory reused sample, no multiplicity/hypothesis-selection correction','CV training sets overlap']},
        'source_prediction_hashes':files}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'monotonicity.json').write_text(json.dumps(monotonic,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k!='source_prediction_hashes'},indent=2))


if __name__=='__main__':main()
