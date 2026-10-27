"""Post-hoc stationary pinhole-prior diagnostic, no learned predictor.

Class priors, yaw zero, stationary motion and 0.5m logistic scale are the same
as the initialized geometry decoder. Only old validation thresholds are fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from cavla.geometric_calibration import optimal_threshold, threshold_metrics
from cavla.geometric_learning import query_curve
from cavla.nuscenes_geometry import rotation, signed_point_box_distance


def object_probabilities(data,geometry,priors):
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((data/'objects.json').read_text());geom={(r['scene'],r['stem']):r for r in json.loads((geometry/'records.json').read_text())}
    probabilities=[]
    for obj in objects:
        r=geom[obj['scene'],obj['stem']];cam=r['input_camera'];b=r['detection_box'];width,length,height=priors[r['label']]
        ray=rotation(cam['rotation']).apply(np.linalg.solve(np.array(cam['camera_intrinsic']),np.array([(b[0]+b[2])/2,(b[1]+b[3])/2,1])))
        depth=np.clip(cam['camera_intrinsic'][1][1]*height/max(b[3]-b[1],1),1,180)
        center=np.array(cam['translation'][:2])+depth*ray[:2];values=[]
        for v,k,h in d['queries']:
            curve=query_curve(float(v)*15,float(k)*.03)
            distance=signed_point_box_distance(curve,np.tile(center,(64,1)),np.zeros(64),np.tile([width,length,height],(64,1)))
            values.append(expit(-(distance[:round(float(h)*6.4*10)].min()-2)/.5))
        probabilities.append(values)
    return d,np.array(probabilities)


def aggregate(d,probabilities,scene_ids,query_ids,keep,method):
    ids=d['scene_object_ids'][scene_ids];mask=keep&(ids>=0)
    p=probabilities[ids.clip(min=0),query_ids[:,None]]
    return np.where(mask,p,0.).max(-1) if method=='prior_max' else 1-np.where(mask,1-p,1.).prod(-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data',type=Path,required=True)
    p.add_argument('--holdout-data',type=Path,required=True)
    p.add_argument('--holdout-geometry',type=Path,required=True)
    p.add_argument('--reference-study',type=Path,required=True)
    p.add_argument('--holdout-study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rm=json.loads((args.reference_study/'manifest.json').read_text());priors=rm['class_priors']
    train,tp=object_probabilities(args.train_data,Path(rm['config']['geometry']),priors)
    hold,hp=object_probabilities(args.holdout_data,args.holdout_geometry,priors)
    objects=json.loads((args.train_data/'objects.json').read_text());scenes=json.loads((args.train_data/'scenes.json').read_text())
    nq=len(train['queries']);group=np.arange(len(scenes)*nq);scene_names=np.repeat(np.array([s['scene'] for s in scenes]),nq)
    hi=np.arange(len(hold['case_scenes'])*nq);hc=hi//nq;hs=hold['case_scenes'][hc];hq=hi%nq
    hcard=hold['case_cardinality'][hc];hlabel=hold['case_targets'].reshape(-1);primary=(hcard>=2)&(hold['query_modes'][hq]=='ood')
    records=[];thresholds={};predictions={};cv_records=[]
    ti=np.arange(len(train['case_scenes'])*nq);tc=ti//nq;ts=train['case_scenes'][tc];tq=ti%nq
    tprimary=(train['case_cardinality'][tc]>=2)&(train['query_modes'][tq]=='ood')
    for fold in range(3):
        split=json.loads((args.reference_study/f'fold{fold}_split.json').read_text());val_names={objects[i]['scene'] for i in split['validation_objects']}
        vi=group[np.isin(scene_names,list(val_names))&np.tile(train['query_modes']=='id',len(scenes))];vs=vi//nq;vq=vi%nq
        vy=train['weak_labels'][vs,vq].reshape(-1);keep=train['scene_object_ids'][vs]>=0
        for method in ['prior_or','prior_max']:
            p0=aggregate(train,tp,vs,vq,keep,method);without=keep.copy();without[np.arange(len(vs)),train['removed_index'][vs]]=False
            p1=aggregate(train,tp,vs,vq,without,method);vp=np.stack([p0,p1],-1)
            threshold,_=optimal_threshold(vy,vp.reshape(-1));prob=aggregate(hold,hp,hs,hq,hold['case_keep'][hc],method)
            use=tprimary&(train['scene_folds'][ts]==fold)
            cv_prob=aggregate(train,tp,ts[use],tq[use],train['case_keep'][tc[use]],method)
            cv_records.append({'method':method,'fold':fold,**threshold_metrics(train['case_targets'].reshape(-1)[use],cv_prob,threshold)})
            result={'method':method,'fold':fold,'threshold':threshold,**threshold_metrics(hlabel[primary],prob[primary],threshold)}
            records.append(result);thresholds[f'fold{fold}_{method}']=threshold;predictions[fold,method]=prob
            np.savez_compressed(args.out/f'fold{fold}_{method}_predictions.npz',indices=hi,probability=prob,threshold=np.array(threshold),
                validation_group_ids=vi,validation_probability=vp)
    # Paired log diagnostic, conditional on fixed predictors/calibrations.
    hold_scenes=json.loads((args.holdout_data/'scenes.json').read_text());logs=np.array([s['log_token'] for s in hold_scenes])[hs[primary]]
    names=sorted(set(logs));rng=np.random.default_rng(20261021);draws=rng.integers(len(names),size=(10000,len(names)))
    weights=np.zeros((10000,len(names)));np.add.at(weights,(np.arange(10000)[:,None],draws),1)
    methods=['prior_or','prior_max','geometry_or','geometry_joint','geometry_joint_metadata'];points={};boots={}
    for method in methods:
        count=np.zeros((len(names),9,4))
        for fold in range(3):
            for sj,seed in enumerate([17,29,43]):
                if method.startswith('prior_'):prob=predictions[fold,method];threshold=thresholds[f'fold{fold}_{method}']
                else:
                    with np.load(args.holdout_study/f'fold{fold}_{method}_seed{seed}_predictions.npz') as z:prob=z['probability'];threshold=float(z['threshold'])
                y=hlabel[primary];pred=prob[primary].astype(float)>=threshold
                for j,log in enumerate(names):
                    use=logs==log;yi=y[use];pi=pred[use]
                    count[j,fold*3+sj]=[(yi&~pi).sum(),(~yi&pi).sum(),yi.sum(),(~yi).sum()]
        pooled=count.sum(0);points[method]=float(((pooled[:,0]/pooled[:,2]+pooled[:,1]/pooled[:,3])/2).mean())
        sample=(weights@count.reshape(len(names),-1)).reshape(10000,9,4)
        valid=np.all(sample[:,:,2:]>0,axis=(1,2));values=np.full(10000,np.nan)
        values[valid]=((sample[valid,:,0]/sample[valid,:,2]+sample[valid,:,1]/sample[valid,:,3])/2).mean(1);boots[method]=values
    comparisons={}
    for baseline in ['prior_or','prior_max']:
        delta=boots['geometry_joint']-boots[baseline];valid=np.isfinite(delta)
        comparisons[baseline]={'difference':points['geometry_joint']-points[baseline],
            'conditional_log_bootstrap_95':np.quantile(delta[valid],[.025,.975]).tolist()}
    summary={'holdout_compound_ood_balanced_error':points,'geometry_joint_minus_prior':comparisons,
        'cv_compound_ood_balanced_error':{m:float(np.mean([r['balanced_error'] for r in cv_records if r['method']==m])) for m in ['prior_or','prior_max']},
        'scope':'Post-hoc diagnostic added after main holdout results. Priors/scale not tuned on holdout; thresholds only old weak validation.',
        'formula':'Stationary box; yaw=0; fixed class sizes; camera ray and pinhole depth; sigmoid(-(min_distance-2)/0.5)',
        'new_trained_models':0,'calibrations':'Three validation thresholds per rule, repeated across seed aliases only to match averaging; not independent trained models',
        'limitations':['Same small 24-scene/13-log compound holdout','Post-hoc control, no selection correction','Fixed model bootstrap, not retraining uncertainty']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'cv_records.json').write_text(json.dumps(cv_records,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
