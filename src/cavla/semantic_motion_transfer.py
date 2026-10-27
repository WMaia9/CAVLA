"""Fixed parked/stopped gating baseline: does semantic readout improve forecasting?

This is a simple diagnostic, not a novel method or a driving-policy evaluation.
All semantic classifiers/thresholds are selected without future-motion labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cavla.temporal_innovation_probe import log_mean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--study',type=Path,default=Path('artifacts/semantic_state_controls_v1_20261021'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    study=args.study
    records=json.loads((study/'records.json').read_text())
    with np.load(study/'predictions.npz') as z:preds={k:z[k] for k in z.files}
    summaries={};saved={};time=np.arange(1,65)*.1
    selected=['logistic_vla','rbf_vla','rbf_numeric','rbf_vla_scene','rbf_clipcrop']
    for split,prefix in [('development','cv'),('previously_used_holdout','holdout')]:
        with np.load(Path('artifacts/semantic_attribute_data_v1_20261021')/f'{split}.npz') as z:
            use=z['slow_vehicle'];d={k:z[k][use] for k in ['numeric','future_centers','logs','attributes','object_indices']}
        center=d['numeric'][:,None,10:12];motion=time[None,:,None]*d['numeric'][:,None,12:14]
        truth=d['future_centers'];cv=center+motion;static=np.broadcast_to(center,truth.shape)
        trajectories={'constant_velocity':[cv],'stationary':[static],
            'oracle_past_attribute':[center+motion*(d['attributes']=='vehicle.stopped')[:,None,None]]}
        for method in selected:
            values=preds[prefix+'_'+method]
            if prefix=='holdout':
                gates=[values[f]>=next(r['threshold'] for r in records if r['fold']==f and r['method']==method and r['evaluation']=='holdout') for f in range(3)]
            else:
                gates=np.zeros(len(values),bool)
                for f in range(3):
                    splitdef=json.loads(Path(f'artifacts/structured_geometry_log_validation_20261021/fold{f}_split.json').read_text())
                    ei=np.isin(d['object_indices'],splitdef['test_objects'])
                    threshold=next(r['threshold'] for r in records if r['fold']==f and r['method']==method and r['evaluation']=='cv')
                    gates[ei]=values[ei]>=threshold
                gates=[gates]
            trajectories[method]=[center+motion*g[:,None,None] for g in gates]
        errors={m:np.mean([np.linalg.norm(t-truth,axis=-1) for t in ts],axis=0) for m,ts in trajectories.items()}
        rows={m:{'log_mean_ade_m':log_mean(e.mean(1),d['logs']), 'log_mean_fde_m':log_mean(e[:,-1],d['logs']),
                 'agent_mean_ade_m':float(e.mean())} for m,e in errors.items()}
        names=sorted(set(d['logs']));rng=np.random.default_rng(20261021)
        draws=rng.integers(len(names),size=(10000,len(names)));comparisons={}
        for baseline in ['constant_velocity','stationary']:
            for method in selected+['oracle_past_attribute']:
                delta=(errors[method]-errors[baseline]).mean(1)
                perlog=np.array([delta[d['logs']==log].mean() for log in names])
                comparisons[method+'_minus_'+baseline]={'ade_difference_m':float(perlog.mean()),
                    'conditional_log_bootstrap_95':np.quantile(perlog[draws].mean(1),[.025,.975]).tolist()}
        summaries[split]={'agents':len(truth),'logs':len(names),'methods':rows,'comparisons':comparisons}
        saved.update({split+'_'+m:e for m,e in errors.items()})
    np.savez_compressed(args.out/'errors.npz',**saved)
    (args.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':'Exploratory fixed downstream forecasting gate; no new learned parameters or driving-policy execution',
        'rule':'If predicted parked, freeze at past-extrapolated current position; otherwise past-only constant velocity',
        'selection':'Frozen existing semantic scores/thresholds selected on attribute validation only; no future-loss tuning',
        'controls':'Always stationary, always constant velocity, true past-attribute gate',
        'limitations':['Previously used evaluation','48 agents,12 logs','Privileged past3D geometry shared by all','Only1 holdout departure','Not a general test of value of semantics']},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({split:{m:r['log_mean_ade_m'] for m,r in s['methods'].items()} for split,s in summaries.items()},indent=2))


if __name__=='__main__':main()
