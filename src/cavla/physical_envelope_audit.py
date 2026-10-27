"""Translation-only sensitivity of logged proximity labels; no safety claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.nuscenes_geometry import signed_point_box_distance

WINDOWS = [(0, 1), (1, 2), (2, 4), (4, 6.4), (0, 6.4)]

def certificates(distance, radius, threshold=2.):
    """Sufficient all-translation certificates, time on final axis."""
    positive = (distance + radius).min(-1) <= threshold
    negative = (distance - radius).min(-1) > threshold
    if np.any(positive & negative):
        raise AssertionError('Inconsistent certificates')
    return positive, negative


def query_curves(queries, times):
    curves = []
    for v, k in queries:
        speed, curvature = float(v)*15, float(k)*.03
        angle = speed*curvature*times
        curves.append(np.stack([speed*times, np.zeros_like(times)], -1) if curvature == 0 else
                      np.stack([np.sin(angle)/curvature, (1-np.cos(angle))/curvature], -1))
    return np.array(curves)


def scene_reduce(values, scenes, positive):
    return np.stack([(values[scenes == s].any(0) if positive else values[scenes == s].all(0))
                     for s in sorted(set(scenes))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    settings = [('development', 'artifacts/partial_composition_data_v2_20261021', 'artifacts/nuscenes_geometry_v3_20261021'),
                ('previously_used_holdout', 'artifacts/composition_holdout_data_v1_20261021', 'artifacts/composition_holdout_geometry_v1_20261021')]
    results = []; replay = {}; times = np.arange(1, 65)*.1
    for tag, objdir, geomdir in settings:
        with np.load(f'artifacts/temporal_necessity_v2_20261021/{tag}_labels_predictions.npz') as z:
            d = {k:z[k] for k in z.files}; use = d['eligible_past']
        objects = json.loads((Path(objdir)/'objects.json').read_text())
        records = {(r['scene'],r['stem']):r for r in json.loads((Path(geomdir)/'records.json').read_text())}
        rr = [records[objects[i]['scene'], objects[i]['stem']] for i in d['object_indices'][use]]
        scenes = d['scenes'][use]; logs = d['logs'][use]
        unique_scenes = sorted(set(scenes)); scene_logs = np.array([logs[scenes == s][0] for s in unique_scenes])
        curves = query_curves(d['queries'], times)
        n = d['numeric'][use]
        for nominal in ['past_cv', 'logged_oracle']:
            centers = n[:,None,10:12]+times[None,:,None]*n[:,None,12:14] if nominal == 'past_cv' else d['future_centers'][use]
            yaws = np.repeat(np.arctan2(n[:,15],n[:,14])[:,None],64,axis=1) if nominal == 'past_cv' else np.array([r['agent_yaws'] for r in rr])
            sizes = np.repeat(n[:,None,16:19],64,axis=1) if nominal == 'past_cv' else np.array([r['agent_sizes'] for r in rr])
            dist = np.stack([np.stack([signed_point_box_distance(q,c,y,s) for q in curves]) for c,y,s in zip(centers,yaws,sizes)])
            original = np.stack([(dist[:,:,(times>lo)&(times<=hi+1e-8)].min(-1)<=2) for lo,hi in WINDOWS],-1)
            expected = d['privileged_past_cv' if nominal == 'past_cv' else 'labels'][use]
            replay[tag+'/'+nominal] = int((original != expected).sum())
            assert np.array_equal(original, expected), replay
            scene_original = scene_reduce(original, scenes, True)
            for acceleration in [0.,.25,.5,1.,2.]:
                # Subset of continuous paths with constant acceleration; not independent time shifts.
                variants = [original]
                if acceleration:
                    for theta in np.arange(8)*np.pi/4:
                        shifted = centers + .5*acceleration*times[None,:,None]**2*np.array([np.cos(theta),np.sin(theta)])
                        dd = np.stack([np.stack([signed_point_box_distance(q,c,y,s) for q in curves]) for c,y,s in zip(shifted,yaws,sizes)])
                        variants.append(np.stack([(dd[:,:,(times>lo)&(times<=hi+1e-8)].min(-1)<=2) for lo,hi in WINDOWS],-1))
                scene_variants = np.stack([scene_reduce(v, scenes, True) for v in variants])
                witnessed = scene_variants.any(0) != scene_variants.all(0)
                for r0 in [0.,.25,.5]:
                    radius = r0+.5*acceleration*times**2
                    pp=[]; nn=[]
                    for lo,hi in WINDOWS:
                        mask=(times>lo)&(times<=hi+1e-8)
                        a,b=certificates(dist[:,:,mask],radius[mask]);pp.append(a);nn.append(b)
                    pos=scene_reduce(np.stack(pp,-1),scenes,True)
                    neg=scene_reduce(np.stack(nn,-1),scenes,False)
                    assert not np.any(pos & ~scene_original) and not np.any(neg & scene_original)
                    assert not np.any(witnessed & (pos|neg))
                    for wi,window in enumerate(WINDOWS):
                        unknown=~(pos[:,:,wi]|neg[:,:,wi])
                        means=lambda v:float(np.mean([v[scene_logs==l].mean() for l in sorted(set(scene_logs))]))
                        results.append(dict(dataset=tag,nominal=nominal,r0_m=r0,acceleration_m_s2=acceleration,
                            window=list(window),agents=len(scenes),scenes=len(unique_scenes),logs=len(set(logs)),
                            inconclusive_fraction=float(unknown.mean()),log_mean_inconclusive_fraction=means(unknown),
                            witnessed_flip_fraction=float(witnessed[:,:,wi].mean()),
                            log_mean_witnessed_flip_fraction=means(witnessed[:,:,wi]),
                            robust_positive_fraction=float(pos[:,:,wi].mean()),robust_negative_fraction=float(neg[:,:,wi].mean())))
            np.savez_compressed(args.out/f'{tag}_{nominal}_nominal.npz',distances=dist,scenes=scenes,logs=logs,queries=d['queries'])
        print(tag, 'complete', flush=True)
    (args.out/'results.json').write_text(json.dumps(results,indent=2))
    (args.out/'manifest.json').write_text(json.dumps(dict(status='EXPLORATORY sensitivity, not calibrated uncertainty or physical counterfactuals',
        replay_label_mismatches=replay,protocol_sha256=hashlib.sha256(Path('docs/PROTOCOLO_FISICA_INFORMACAO_V1.md').read_bytes()).hexdigest(),
        caveats=['Previously examined data','Translation only; fixed nominal yaw/size','Generated constant acceleration paths need not be road-feasible',
                 'Certificate failure is not necessarily attainable ambiguity','No learned model, reactive agent, ego footprint or action selection']),indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())

if __name__=='__main__':main()
