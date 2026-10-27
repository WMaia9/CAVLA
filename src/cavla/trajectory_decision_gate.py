"""Controller-free decision gate: do solver changes flip trajectory-level decisions that mean ADE hides?

Decision classes are read from the 6.4s plan with thresholds fixed before results
(docs/PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md, adendo 3). The branch margin is
measured on the cheap Euler10 plan, so it is available at inference time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.native_command_gate_multi import auroc, episode_bootstrap

DT=.1;STOP_EXTENT_M=1.5;HEADING_DEG=20.;OFFSET_M=1.75


def plan_speed(traj,start=20,stop=30):
    return float(np.linalg.norm(np.diff(traj[start-1:stop,:2],axis=0),axis=1).mean()/DT)


def decisions(traj,current_speed):
    traj=np.asarray(traj,float);extent=float(np.linalg.norm(traj[:,:2],axis=1).max())
    speed=plan_speed(traj);band=max(1.,.15*current_speed)
    if extent<=STOP_EXTENT_M:longitudinal='stop'
    elif speed<current_speed-band:longitudinal='decelerate'
    elif speed>current_speed+band:longitudinal='accelerate'
    else:longitudinal='keep'
    tail=traj[-1,:2]-traj[-6,:2];heading=float(np.degrees(np.arctan2(tail[1],tail[0]))) if np.linalg.norm(tail)>.2 else 0.
    offset=float(traj[-1,1])
    if heading>HEADING_DEG:lateral='left'
    elif heading<-HEADING_DEG:lateral='right'
    elif offset>OFFSET_M:lateral='shift_left'
    elif offset<-OFFSET_M:lateral='shift_right'
    else:lateral='straight'
    # Distance to the nearest class boundary, normalized by each threshold scale.
    long_margin=min(abs(extent-STOP_EXTENT_M)/STOP_EXTENT_M,abs(abs(speed-current_speed)-band)/band)
    lat_margin=min(abs(abs(heading)-HEADING_DEG)/HEADING_DEG,abs(abs(offset)-OFFSET_M)/OFFSET_M)
    return {'longitudinal':longitudinal,'lateral':lateral,'plan_speed':speed,'extent':extent,'heading_deg':heading,
        'offset_m':offset,'long_margin':long_margin,'lat_margin':lat_margin}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    contexts={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    with np.load(args.predictions/'predictions.npz') as z:arrays={k:z[k] for k in z.files}
    rows=[]
    for name,context in sorted(contexts.items()):
        group=context.get('episode',context.get('clip_id'));speed=context['speed_mps']
        with np.load(args.data/f'{name}.npz') as z:human=z['ego_future_xyz'] if 'ego_future_xyz' in z.files else None
        for seed in [9201,9202]:
            prefix=f'{name}_s{seed}_';plans={k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix) and v.shape==(64,3)}
            v=arrays[prefix+'euler10_velocities'].reshape(10,-1)
            cos=np.sum(v[1:]*v[:-1],axis=1)/(np.linalg.norm(v[1:],axis=1)*np.linalg.norm(v[:-1],axis=1))
            cheap=decisions(plans['euler10'],speed)
            signals={'initial_cosine':-float(cos[0]),'min_cosine':-float(cos.min()),
                'curvature':float(np.linalg.norm(v[1:]-v[:-1],axis=1).sum()/np.linalg.norm(v,axis=1).mean()),
                'long_branch_margin':-cheap['long_margin'],'lat_branch_margin':-cheap['lat_margin']}
            classes={solver:decisions(plan,speed) for solver,plan in plans.items()}
            human_class=decisions(human,speed) if human is not None else None
            for reference in ['heun160','euler10']:
                for solver,plan in plans.items():
                    if solver==reference:continue
                    a,b=classes[solver],classes[reference]
                    row={'context':name,'group':group,'seed':seed,'reference':reference,'solver':solver,
                        'long_flip':a['longitudinal']!=b['longitudinal'],'lat_flip':a['lateral']!=b['lateral'],
                        'ade_to_reference':float(np.linalg.norm(plan[:,:2]-plans[reference][:,:2],axis=1).mean()),
                        'reference_long':b['longitudinal'],'solver_long':a['longitudinal'],
                        'reference_lat':b['lateral'],'solver_lat':a['lateral'],**signals}
                    if human is not None:
                        row['ade_to_human']=float(np.linalg.norm(plan[:,:2]-human[:,:2],axis=1).mean())
                        row['reference_ade_to_human']=float(np.linalg.norm(plans[reference][:,:2]-human[:,:2],axis=1).mean())
                        row['long_agrees_human']=a['longitudinal']==human_class['longitudinal']
                        row['reference_long_agrees_human']=b['longitudinal']==human_class['longitudinal']
                        row['lat_agrees_human']=a['lateral']==human_class['lateral']
                        row['reference_lat_agrees_human']=b['lateral']==human_class['lateral']
                    rows.append(row)
    rng=np.random.default_rng(20261022);summary=[]
    for reference,solver in sorted({(r['reference'],r['solver']) for r in rows}):
        sub=[r for r in rows if r['reference']==reference and r['solver']==solver];groups=[r['group'] for r in sub]
        decision=[float(r['long_flip'] or r['lat_flip']) for r in sub];ade=np.array([r['ade_to_reference'] for r in sub])
        entry={'reference':reference,'solver':solver,'n':len(sub),'groups':len(set(groups)),
            'long_flip_rate':float(np.mean([r['long_flip'] for r in sub])),'lat_flip_rate':float(np.mean([r['lat_flip'] for r in sub])),
            'any_flip_rate':float(np.mean(decision)),'any_flip_ci':episode_bootstrap(decision,groups,rng),
            'ade_mean':float(ade.mean()),'ade_median':float(np.median(ade)),'ade_p95':float(np.quantile(ade,.95)),'ade_max':float(ade.max())}
        for signal in ['initial_cosine','min_cosine','curvature','long_branch_margin','lat_branch_margin','ade_to_reference']:
            entry[f'auroc_any_flip_{signal}']=auroc([r[signal] for r in sub],decision)
        if 'ade_to_human' in sub[0]:
            delta=[r['ade_to_human']-r['reference_ade_to_human'] for r in sub]
            entry.update({'mean_ade_to_human':float(np.mean([r['ade_to_human'] for r in sub])),
                'reference_mean_ade_to_human':float(np.mean([r['reference_ade_to_human'] for r in sub])),
                'delta_ade_to_human':float(np.mean(delta)),'delta_ade_to_human_ci':episode_bootstrap(delta,groups,rng),
                'long_agree_human':float(np.mean([r['long_agrees_human'] for r in sub])),
                'reference_long_agree_human':float(np.mean([r['reference_long_agrees_human'] for r in sub])),
                'lat_agree_human':float(np.mean([r['lat_agrees_human'] for r in sub])),
                'reference_lat_agree_human':float(np.mean([r['reference_lat_agrees_human'] for r in sub]))})
        summary.append(entry)
    result={'scope':__doc__,'thresholds':{'stop_extent_m':STOP_EXTENT_M,'heading_deg':HEADING_DEG,'offset_m':OFFSET_M,
        'speed_band':'max(1 m/s, 15% current speed), plan speed over 2-3 s'},'summary':summary,'rows':rows,
        'limitations':['Decision classes are coarse thresholds on the plan, not executed behavior',
            'Heun160 is a numerical reference; human future is one logged trajectory','Bootstrap groups are clips (PhysicalAI) or episodes (CARLA)']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),
        'predictions_manifest_sha256':hashlib.sha256((args.predictions/'manifest.json').read_bytes()).hexdigest()},indent=2))
    keys=['reference','solver','n','long_flip_rate','lat_flip_rate','any_flip_ci','ade_mean','ade_p95','auroc_any_flip_initial_cosine',
        'auroc_any_flip_long_branch_margin','delta_ade_to_human','long_agree_human','reference_long_agree_human']
    for entry in summary:print(json.dumps({k:entry.get(k) for k in keys}),flush=True)


if __name__=='__main__':main()
