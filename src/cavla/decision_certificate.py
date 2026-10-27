"""Zero-extra-evaluation decision certificates from the Euler10 integration already computed.

Classic a posteriori ingredients (defect/trapezoid correction from consecutive
velocities, endpoint predictions x+(1-t)v) are mapped through the decision
functional. Question: does estimated decision shift / branch margin predict
flips against the converged flow better than the margin alone, and does the
free correction itself change decisions toward the converged flow?
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from cavla.native_command_gate_multi import auroc, episode_bootstrap
from cavla.trajectory_decision_gate import decisions, STOP_EXTENT_M, HEADING_DEG, OFFSET_M

CHECKPOINT=Path('/home/perceptron-lab/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/7aba8293c09993f2e125c6819df05d7fa3e873ea')


def defect_corrected(states,velocities,dt):
    """Trapezoid correction of explicit Euler using only evaluated velocities; last slope extrapolated linearly."""
    x=states[-1]+dt*velocities[-1];slopes=list(velocities[1:])+[2*velocities[-1]-velocities[-2]]
    return x+sum(.5*dt*(nxt-cur) for cur,nxt in zip(velocities,slopes))


def endpoint_predictions(states,velocities,times):
    return [s+(1-t)*v for s,v,t in zip(states,velocities,times)]


def normalized_shift(a,b,speed):
    band=max(1.,.15*speed)
    return max(abs(a['extent']-b['extent'])/STOP_EXTENT_M,abs(a['plan_speed']-b['plan_speed'])/band,
        abs(a['heading_deg']-b['heading_deg'])/HEADING_DEG,abs(a['offset_m']-b['offset_m'])/OFFSET_M)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    sys.path.insert(0,'/mnt/datasets/alpamayo/carlamayo-integration/third_party/alpamayo1.5/src')
    from alpamayo1_5.action_space import UnicycleAccelCurvatureActionSpace
    cfg=json.loads((CHECKPOINT/'config.json').read_text())['action_space_cfg'];cfg.pop('_target_')
    space=UnicycleAccelCurvatureActionSpace(**cfg).eval()
    contexts={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    with np.load(args.predictions/'predictions.npz') as z:arrays={k:z[k] for k in z.files}
    rows=[];replay=[]
    for name,context in sorted(contexts.items()):
        group=context.get('episode',context.get('clip_id'));speed=context['speed_mps']
        with np.load(args.data/f'{name}.npz') as z:hx=torch.from_numpy(z['ego_history_xyz']).float()[None];hr=torch.from_numpy(z['ego_history_rot']).float()[None]
        def traj(action):
            with torch.no_grad():xyz,_=space.action_to_traj(torch.from_numpy(np.asarray(action,np.float32)).reshape(1,*arrays[f'{name}_s9201_euler10_action'].shape[1:]),hx,hr)
            return xyz.reshape(-1,3).numpy()
        for seed in [9201,9202]:
            prefix=f'{name}_s{seed}_';states=arrays[prefix+'euler10_states'];velocities=arrays[prefix+'euler10_velocities']
            times=np.linspace(0,1,11)[:10]
            euler=traj(arrays[prefix+'euler10_action']);replay.append(float(np.abs(euler-arrays[prefix+'euler10']).max()))
            corrected=traj(defect_corrected(states,velocities,.1))
            endpoints=[traj(e) for e in endpoint_predictions(states,velocities,times)[-3:]]
            ref=arrays[prefix+'heun160']
            d_e,d_c,d_r=decisions(euler,speed),decisions(corrected,speed),decisions(ref,speed)
            d_end=[decisions(e,speed) for e in endpoints]
            key=lambda d:(d['longitudinal'],d['lateral'])
            margin=min(d_e['long_margin'],d_e['lat_margin']);shift=normalized_shift(d_c,d_e,speed)
            rows.append({'context':name,'group':group,'seed':seed,
                'flip_euler10':key(d_e)!=key(d_r),'flip_corrected':key(d_c)!=key(d_r),
                'ade_euler10':float(np.linalg.norm(euler[:,:2]-ref[:,:2],axis=1).mean()),
                'ade_corrected':float(np.linalg.norm(corrected[:,:2]-ref[:,:2],axis=1).mean()),
                'score_margin':-margin,'score_shift':shift,'score_ratio':shift/max(margin,1e-6),
                'score_corrected_disagrees':float(key(d_c)!=key(d_e))+shift/max(margin,1e-6)*1e-3,
                'score_endpoint_unstable':float(len({key(d) for d in d_end+[d_e]})>1)-margin*1e-3})
    flips=[r['flip_euler10'] for r in rows];groups=[r['group'] for r in rows];rng=np.random.default_rng(20261022)
    summary={'n':len(rows),'groups':len(set(groups)),'max_replay_error_m':max(replay),
        'flip_rate_euler10':float(np.mean(flips)),'flip_rate_corrected':float(np.mean([r['flip_corrected'] for r in rows])),
        'flip_rate_corrected_minus_euler10_ci':episode_bootstrap([float(r['flip_corrected'])-float(r['flip_euler10']) for r in rows],groups,rng),
        'mean_ade_euler10':float(np.mean([r['ade_euler10'] for r in rows])),'mean_ade_corrected':float(np.mean([r['ade_corrected'] for r in rows])),
        **{f'auroc_{s}':auroc([r[f'score_{s}'] for r in rows],flips) for s in ['margin','shift','ratio','corrected_disagrees','endpoint_unstable']}}
    for budget in [.1,.2,.3]:
        for s in ['margin','ratio','shift']:
            score=np.array([r[f'score_{s}'] for r in rows]);chosen=score>=np.quantile(score,1-budget)
            summary[f'recall_at_{int(budget*100)}pct_refine_{s}']=float(chosen[np.array(flips)].mean()) if any(flips) else None
    (args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'summary':summary,'rows':rows},indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),
        'predictions_manifest_sha256':hashlib.sha256((args.predictions/'manifest.json').read_bytes()).hexdigest()},indent=2))
    print(json.dumps(summary,indent=1))


if __name__=='__main__':main()
