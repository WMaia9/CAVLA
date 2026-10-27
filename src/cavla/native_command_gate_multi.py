"""Multi-episode offline gate: does solver error reach PID commands, and do cheap signals predict it?

Common declared PID state (empty buffers, previous steer 0, recorded speed). No
closed-loop world is run. Numerical references are not correct actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr


def mode(out):
    if out['brake']>1e-6:return 'brake'
    if out['throttle']>1e-6:return 'throttle'
    return 'coast'


def auroc(score,label):
    score=np.asarray(score,float);label=np.asarray(label,bool)
    if label.all() or (~label).all():return None
    pos=score[label];neg=score[~label]
    return float((pos[:,None]>neg[None]).mean()+.5*(pos[:,None]==neg[None]).mean())


def episode_bootstrap(values,episodes,rng,reps=4000):
    values=np.asarray(values,float);episodes=np.asarray(episodes);unique=np.unique(episodes)
    groups=[values[episodes==e] for e in unique];stats=[]
    for _ in range(reps):
        pick=rng.integers(len(unique),size=len(unique));stats.append(np.concatenate([groups[i] for i in pick]).mean())
    return [float(np.quantile(stats,.025)),float(np.quantile(stats,.975))]


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    root=Path('/mnt/datasets/alpamayo/carlamayo-integration');sys.path.insert(0,str(root))
    sys.path.append('/home/perceptron-lab/CARLA_0.9.16/PythonAPI/carla')
    from module.pid_controller import OfficialPIDFollower
    contexts={r['name']:r for r in json.loads((args.data/'records.json').read_text()) if 'error' not in r}
    for record in contexts.values():
        record.setdefault('episode',record.get('clip_id'));record.setdefault('map',record.get('country'))
    with np.load(args.predictions/'predictions.npz') as z:arrays={key:z[key] for key in z.files}
    class Vehicle:
        def __init__(self,speed):self.speed=speed
        def get_world(self):return None
        def get_transform(self):return carla.Transform()
        def get_velocity(self):return carla.Vector3D(x=self.speed)
        def get_control(self):return carla.VehicleControl(steer=0.)
    def command(pred,speed):
        vehicle=Vehicle(speed);controller=OfficialPIDFollower(None,vehicle)
        steer,throttle,brake,debug=controller.compute_control(vehicle.get_transform(),pred,speed)
        return {'steer':float(steer),'throttle':float(throttle),'brake':float(brake),
            'commanded_mps':float(debug.get('commanded_mps',0.)),'target_idx':int(debug.get('target_idx',-1)),'pid_branch':debug['mode']}
    rows=[]
    for name,context in sorted(contexts.items()):
        for seed in [9201,9202]:
            prefix=f'{name}_s{seed}_';solvers={k[len(prefix):]:v for k,v in arrays.items() if k.startswith(prefix) and v.shape==(64,3)}
            if 'heun160' not in solvers or 'euler10' not in solvers:raise RuntimeError(f'Missing reference for {prefix}')
            with np.load(args.data/f'{name}.npz') as z:
                if 'ego_future_xyz' in z.files:solvers['human']=z['ego_future_xyz'].astype(np.float32)
            v=arrays[prefix+'euler10_velocities'].reshape(10,-1)
            cos=np.sum(v[1:]*v[:-1],axis=1)/(np.linalg.norm(v[1:],axis=1)*np.linalg.norm(v[:-1],axis=1))
            cheap={'initial_cosine':float(cos[0]),'min_cosine':float(cos.min()),'initial_field_norm':float(np.linalg.norm(v[0])),
                'speed_mps':context['speed_mps']}
            outputs={solver:command(pred,context['speed_mps']) for solver,pred in solvers.items()}
            for reference in [r for r in ['heun160','euler10','human'] if r in outputs]:
                ref=outputs[reference];ref_traj=solvers[reference]
                for solver,out in outputs.items():
                    if solver==reference or solver=='human':continue
                    rows.append({'context':name,'episode':context['episode'],'map':context['map'],'seed':seed,
                        'reference':reference,'solver':solver,'mode':mode(out),'reference_mode':mode(ref),
                        'mode_flip':mode(out)!=mode(ref),'abs_steer_error':abs(out['steer']-ref['steer']),
                        'abs_throttle_minus_brake_error':abs((out['throttle']-out['brake'])-(ref['throttle']-ref['brake'])),
                        'abs_commanded_speed_error':abs(out['commanded_mps']-ref['commanded_mps']),
                        'target_idx_change':out['target_idx']!=ref['target_idx'],'pid_branch_change':out['pid_branch']!=ref['pid_branch'],
                        'mean_xy_m':float(np.linalg.norm(solvers[solver][:,:2]-ref_traj[:,:2],axis=-1).mean()),
                        'first_second_xy_m':float(np.linalg.norm(solvers[solver][:10,:2]-ref_traj[:10,:2],axis=-1).mean()),
                        **cheap,'command':out,'reference_command':ref})
    rng=np.random.default_rng(20261022);summary=[]
    pairs=sorted({(r['reference'],r['solver']) for r in rows})
    for reference,solver in pairs:
        sub=[r for r in rows if r['reference']==reference and r['solver']==solver];ep=[r['episode'] for r in sub]
        flips=[float(r['mode_flip']) for r in sub];steer=[float(r['abs_steer_error']>.05) for r in sub]
        speed=[r['abs_commanded_speed_error'] for r in sub];action=[r['abs_throttle_minus_brake_error'] for r in sub]
        entry={'reference':reference,'solver':solver,'n':len(sub),'episodes':len(set(ep)),
            'mode_flip_rate':float(np.mean(flips)),'mode_flip_ci':episode_bootstrap(flips,ep,rng),
            'steer_gt_005_rate':float(np.mean(steer)),'steer_gt_005_ci':episode_bootstrap(steer,ep,rng),
            'median_commanded_speed_error':float(np.median(speed)),'mean_commanded_speed_error':float(np.mean(speed)),
            'mean_long_action_error':float(np.mean(action)),'mean_long_action_ci':episode_bootstrap(action,ep,rng)}
        for predictor in ['initial_cosine','min_cosine','initial_field_norm','speed_mps','mean_xy_m','first_second_xy_m']:
            x=np.array([r[predictor] for r in sub]);sign=-1 if 'cosine' in predictor else 1
            rho=spearmanr(sign*x,action).statistic
            entry[f'spearman_long_action_vs_{predictor}']=None if np.isnan(rho) else float(rho)
            entry[f'auroc_flip_vs_{predictor}']=auroc(sign*x,[r['mode_flip'] for r in sub])
        summary.append(entry)
    result={'scope':__doc__,'contexts':len(contexts),'episodes':len({c['episode'] for c in contexts.values()}),
        'summary':summary,'rows':rows,'predictor_sign':'Cosines negated so larger means more expected error',
        'limitations':['Common empty PID buffers and previous steer 0, not recorded controller memory',
            'Scripted lane-following CARLA episodes, four maps, static NPCs','Two contexts per episode are correlated; CIs bootstrap episodes',
            'Heun160 is an approximate numerical reference, not a correct action','No vehicle execution or safety assessment']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    files=[root/'module/pid_controller.py',root/'module/config.py']
    (args.out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source).hexdigest(),
        'predictions_manifest_sha256':hashlib.sha256((args.predictions/'manifest.json').read_bytes()).hexdigest(),
        'external_sources_read_only_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2))
    for entry in summary:print(json.dumps({k:entry[k] for k in ['reference','solver','n','mode_flip_rate','mode_flip_ci',
        'steer_gt_005_rate','mean_long_action_error','spearman_long_action_vs_initial_cosine','auroc_flip_vs_initial_cosine',
        'auroc_flip_vs_mean_xy_m']}),flush=True)


if __name__=='__main__':main()
