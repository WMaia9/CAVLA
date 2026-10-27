"""Compose previously fitted individually supervised predictors as references.

These models had stronger training and checkpoint labels; they are not a
matched-supervision baseline or a guaranteed performance upper bound.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cavla.geometric_calibration import optimal_threshold, threshold_metrics
from cavla.geometric_learning import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--object-calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    with np.load(args.data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((args.data/'objects.json').read_text());nq=len(d['queries'])
    parent=Path(json.loads((args.data/'manifest.json').read_text())['parent'])
    with np.load(parent/'dataset.npz') as z:object_ids=z['object_ids']
    assert np.array_equal(object_ids,np.repeat(np.arange(len(objects)),nq))
    study=Path(json.loads((args.object_calibration/'manifest.json').read_text())['study'])
    case=np.arange(len(d['case_scenes'])*nq)//nq;qi=np.arange(len(case))%nq;si=d['case_scenes'][case]
    card=d['case_cardinality'][case];target=d['case_targets'].reshape(-1)
    ambiguous=(d['oracle_lower']!=d['oracle_upper']).reshape(-1);rows=[];hashes={}
    safe=d['scene_object_ids'].clip(min=0);present=d['scene_object_ids']>=0
    for fold in range(3):
        split=json.loads((study/f'fold{fold}_split.json').read_text())
        val_scenes=np.isin(d['scene_object_ids'][:,0],split['validation_objects'])
        vgroups=np.flatnonzero(np.repeat(val_scenes,nq)&np.tile(d['query_modes']=='id',len(safe)))
        vs=vgroups//nq;vq=vgroups%nq;vy=d['weak_labels'][vs,vq].reshape(-1)
        ti=np.flatnonzero(d['scene_folds'][si]==fold)
        for seed in [17,29,43]:
            path=args.object_calibration/f'fold{fold}_geometry_spatial_seed{seed}_predictions.npz'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            matrix=np.full(len(objects)*nq,np.nan)
            with np.load(path) as z:
                matrix[z['indices']]=z['probability'];matrix[z['validation_indices']]=z['validation_probability']
            matrix=matrix.reshape(len(objects),nq)
            def aggregate(scene_index,query_index,keep,method):
                p=matrix[safe[scene_index],query_index[:,None]];mask=keep&present[scene_index]
                assert np.isfinite(p[mask]).all()
                if method=='supervised_max':return np.where(mask,p,0.).max(-1)
                return 1-np.where(mask,1-p,1.).prod(-1)
            for method in ['supervised_or','supervised_max']:
                keep=present[vs].copy();v0=aggregate(vs,vq,keep,method)
                keep[np.arange(len(vs)),d['removed_index'][vs]]=False;v1=aggregate(vs,vq,keep,method)
                vp=np.stack([v0,v1],-1);threshold,_=optimal_threshold(vy,vp.reshape(-1))
                probability=aggregate(si[ti],qi[ti],d['case_keep'][case[ti]],method)
                masks={'original':card[ti]==0,'single':card[ti]==1,'double':card[ti]==2,'triple':card[ti]==3,
                    'compound':card[ti]>=2,'compound_ambiguous':(card[ti]>=2)&ambiguous[ti],
                    'compound_identified':(card[ti]>=2)&~ambiguous[ti]}
                for view,mask in masks.items():
                    for mode in ['id','ood']:
                        use=mask&(d['query_modes'][qi[ti]]==mode)
                        if not use.any():continue
                        rows.append({'method':method,'fold':fold,'seed':seed,'view':view,'mode':mode,
                            **threshold_metrics(target[ti][use],probability[use],threshold),
                            'fixed_threshold_balanced_error':metrics(target[ti][use],probability[use])['balanced_error'],
                            'threshold':threshold})
                np.savez_compressed(args.out/f'fold{fold}_{method}_seed{seed}_predictions.npz',indices=ti,
                    probability=probability,threshold=np.array(threshold),validation_group_ids=vgroups,validation_probability=vp)
    def mean_field(method,view,mode,key):
        values=[r[key] for r in rows if r['method']==method and r['view']==view and r['mode']==mode and r[key] is not None]
        return float(np.mean(values)) if values else None
    summary={m:{v:{mode:{k:mean_field(m,v,mode,k) for k in ['auc','balanced_error','false_negative_rate','false_positive_rate',
        'fixed_threshold_balanced_error']} for mode in ['id','ood']} for v in masks} for m in ['supervised_or','supervised_max']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'results.json').write_text(json.dumps(rows,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':'Stronger supervision reference; reuses nine fitted models, not new training',
        'data':str(args.data),'object_calibration':str(args.object_calibration),'source_hashes':hashes,
        'selection':'Original checkpoints selected using individual validation targets; aggregate threshold calibrated on weak validation pairs'},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({m:summary[m]['compound']['ood'] for m in summary},indent=2))


if __name__=='__main__':main()
