"""Numeric ego-history controls matched to the four observed camera timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.nuscenes_geometry import iter_array,rotation


def ego_features(times,positions,quaternions):
    times=np.asarray(times,np.int64);positions=np.asarray(positions,float)
    assert np.all(np.diff(times)>0)
    inv=rotation(quaternions[-1]).inv();relative=inv.apply(positions-positions[-1])
    yaw=np.array([(inv*rotation(q)).as_euler('xyz')[2] for q in quaternions])
    dt=(times-times[-1])*1e-6;velocity=np.diff(relative,axis=0)/np.diff(dt)[:,None]
    return np.r_[relative[:-1].ravel(),np.sin(yaw[:-1]),np.cos(yaw[:-1]),dt[:-1],velocity.ravel()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=Path('artifacts/front_transition_data_v1_20261021');rows=json.loads((source/'objects.json').read_text())
    histories=json.loads(Path('artifacts/transition_visual_history_data_v1_20261021/histories.json').read_text())
    tokens={f['token'] for history in histories.values() for f in history};tables=Path('/mnt/datasets/nuScenes/v1.0-trainval')
    cameras={r['token']:r for r in iter_array(tables/'sample_data.json') if r['token'] in tokens}
    wanted={c['ego_pose_token'] for c in cameras.values()};egos={r['token']:r for r in iter_array(tables/'ego_pose.json') if r['token'] in wanted}
    features={}
    for image,history in histories.items():
        poses=[egos[cameras[f['token']]['ego_pose_token']] for f in history]
        # Ego pose timestamps can be sensor-specific; use the actual pose timestamps for velocities.
        times=[p['timestamp'] for p in poses]
        assert all(abs(t-f['timestamp'])<=100000 for t,f in zip(times,history))
        assert max(times)<=history[-1]['timestamp']+100000
        # Reject future ego poses relative to the camera: no optimistic tolerance.
        assert all(t<=history[-1]['timestamp'] for t in times)
        features[image]=ego_features(times,[p['translation'] for p in poses],[p['rotation'] for p in poses])
    extra=np.array([features[r['image']] for r in rows])
    with np.load(source/'features.npz') as z:arrays={k:z[k] for k in z.files}
    arrays['relative_geometry']=np.c_[arrays['relative_geometry'],extra]
    np.savez_compressed(args.out/'features.npz',**arrays);(args.out/'objects.json').write_bytes((source/'objects.json').read_bytes())
    (args.out/'manifest.json').write_text(json.dumps({'objects_sha256':hashlib.sha256((source/'objects.json').read_bytes()).hexdigest(),
        'parent':str(source),'ego_history_dimensions':int(extra.shape[1]),'history_images':len(histories),
        'scope':'Four past/current ego poses as control; no labels or images altered; all pose times <= current camera'},indent=2))
    print('complete',extra.shape,flush=True)


if __name__=='__main__':main()
