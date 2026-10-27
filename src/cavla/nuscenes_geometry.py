"""Recover 3D annotation candidates and recorded agent futures for local edits.

Automatic 2D/3D joins are candidates, not verified removal identities. Logged
agent futures do not model reactions to counterfactual ego actions or deletions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from cavla.real_pair_alignment import unique_box


def iter_array(path, chunk_size=4*1024*1024):
    """Stream a JSON array without loading multi-GB nuScenes tables."""
    decoder=json.JSONDecoder();buffer='';pos=0;eof=False;started=False
    with Path(path).open() as stream:
        while True:
            while pos<len(buffer) and buffer[pos] in ' \n\r\t,':pos+=1
            if not started and pos<len(buffer):
                if buffer[pos]!='[':raise ValueError('Expected JSON array')
                pos+=1;started=True;continue
            if pos<len(buffer) and buffer[pos]==']':return
            if pos<len(buffer):
                try: value,end=decoder.raw_decode(buffer,pos)
                except json.JSONDecodeError:
                    if eof:raise
                else:
                    pos=end;yield value;continue
            if eof:raise ValueError('Unterminated JSON array')
            buffer=buffer[pos:]+stream.read(chunk_size);pos=0
            if len(buffer)==0:raise ValueError('Empty or truncated array')
            if stream.tell()==Path(path).stat().st_size:eof=True


def rotation(q):
    return Rotation.from_quat([q[1],q[2],q[3],q[0]])


def iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    intersection=float(np.maximum(hi-lo,0).prod())
    union=float(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-intersection)
    return intersection/max(union,1e-12)


def category_label(name):
    if name.startswith('human.pedestrian'):return 'person'
    for label in ['car','truck','bus','bicycle','motorcycle']:
        if name=='vehicle.'+label or name.startswith('vehicle.'+label+'.'):return label
    return None


def projected_box(annotation,ego,sensor,width=1600,height=900):
    w,l,h=annotation['size']
    corners=np.array([[x*l/2,y*w/2,z*h/2] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    world=rotation(annotation['rotation']).apply(corners)+annotation['translation']
    vehicle=rotation(ego['rotation']).inv().apply(world-ego['translation'])
    camera=rotation(sensor['rotation']).inv().apply(vehicle-sensor['translation'])
    if np.any(camera[:,2]<=.1):return None # Do not invent near-plane clipping.
    projected=camera@np.array(sensor['camera_intrinsic']).T
    pixels=projected[:,:2]/projected[:,2,None]
    lo=np.maximum(pixels.min(0),[0,0]);hi=np.minimum(pixels.max(0),[width,height])
    if np.any(hi<=lo):return None
    return np.r_[lo,hi].tolist()


def signed_point_box_distance(points,centers,yaws,sizes):
    """Point centerline to oriented rectangle, excludes ego footprint."""
    delta=points-centers
    c=np.cos(yaws);s=np.sin(yaws)
    local=np.stack([c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]],-1)
    half=sizes[:,[1,0]]/2 # nuScenes size order width,length,height
    q=np.abs(local)-half
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nuscenes',type=Path,default=Path('/mnt/datasets/nuScenes'))
    p.add_argument('--pairs',type=Path,default=Path('artifacts/real_audit_20261020/pairs.csv'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen')
    tables=args.nuscenes/'v1.0-trainval'
    ranked={r['scene_name']:r for r in csv.DictReader((root/'scenes_ranked.csv').open())}
    pairs=list(csv.DictReader(args.pairs.open()))
    selected={r['scene'] for r in pairs};ranked={s:ranked[s] for s in selected}
    samples={r['token']:r for r in json.loads((tables/'sample.json').read_text())}
    selected_scene_tokens={samples[r['sample_token']]['scene_token'] for r in ranked.values()}
    wanted_samples={t for t,r in samples.items() if r['scene_token'] in selected_scene_tokens}
    wanted_sd={r['sd_token'] for r in ranked.values()}
    print('Scanning sample_data',flush=True)
    sd={r['token']:r for r in iter_array(tables/'sample_data.json') if r['token'] in wanted_sd}
    wanted_ego={r['ego_pose_token'] for r in sd.values()}
    print('Scanning ego_pose',flush=True)
    egos={r['token']:r for r in iter_array(tables/'ego_pose.json') if r['token'] in wanted_ego}
    sensors={r['token']:r for r in json.loads((tables/'calibrated_sensor.json').read_text())}
    instances={r['token']:r for r in json.loads((tables/'instance.json').read_text())}
    categories={r['token']:r['name'] for r in json.loads((tables/'category.json').read_text())}
    print('Scanning annotations for selected scenes',flush=True)
    annotations={r['token']:r for r in iter_array(tables/'sample_annotation.json') if r['sample_token'] in wanted_samples}
    by_sample={}
    for r in annotations.values():by_sample.setdefault(r['sample_token'],[]).append(r)
    results=[]
    for scene in sorted(selected):
        meta=ranked[scene];sample=samples[meta['sample_token']];sample_data=sd[meta['sd_token']]
        ego=egos[sample_data['ego_pose_token']];sensor=sensors[sample_data['calibrated_sensor_token']]
        folder=root/'nuscenes_per_scene'/scene
        detections=json.loads((folder/f'{scene}_seg.json').read_text())
        projected=[]
        for ann in by_sample.get(meta['sample_token'],[]):
            category=categories[instances[ann['instance_token']]['category_token']]
            box=projected_box(ann,ego,sensor)
            if box is not None:projected.append((ann,category,box))
        with np.load(folder/f'{scene}_data.npz') as z:future=z['future_xyz'][:,:2].astype(float)
        for pair in [r for r in pairs if r['scene']==scene]:
            stem=pair['stem'];box=unique_box(stem,detections,1600,900)
            record={'scene':scene,'stem':stem,'status':'no_unique_detection_box'}
            if box is not None:
                label=stem.split('_',1)[1].rsplit('_conf',1)[0]
                candidates=sorted([(iou(box,b),ann,cat,b) for ann,cat,b in projected if category_label(cat)==label],key=lambda x:x[0],reverse=True)
                best=candidates[0][0] if candidates else 0.;second=candidates[1][0] if len(candidates)>1 else 0.
                record.update(label=label,best_iou=best,runner_up_iou=second,
                              status='ambiguous_or_low_iou',front_sample_offset_seconds=(sample_data['timestamp']-sample['timestamp'])*1e-6)
                if best>=.5 and best-second>=.15:
                    _,ann,category,projected_bbox=candidates[0]
                    # Recover neighboring logged annotations, with no extrapolation.
                    track=[ann]
                    if ann['prev'] in annotations:track.insert(0,annotations[ann['prev']])
                    cursor=ann
                    for _ in range(15):
                        if cursor['next'] not in annotations:break
                        cursor=annotations[cursor['next']];track.append(cursor)
                    times=np.array([samples[a['sample_token']]['timestamp'] for a in track],dtype=float)*1e-6
                    grid=sample_data['timestamp']*1e-6+np.arange(1,65)*.1
                    valid=(grid>=times.min())&(grid<=times.max())
                    if len(times)>1:
                        brackets=np.clip(np.searchsorted(times,grid,side='right'),1,len(times)-1)
                        valid &= (times[brackets]-times[brackets-1])<=.75
                    record.update(status='automatic_3d_candidate',annotation_token=ann['token'],instance_token=ann['instance_token'],
                                  category=category,detection_box=box,projected_box=projected_bbox,valid_future_steps=int(valid.sum()),
                                  input_camera={k:sensor[k] for k in ['camera_intrinsic','rotation','translation']})
                    if len(track)>1 and valid.any():
                        points=np.array([a['translation'] for a in track]);sizes=np.array([a['size'] for a in track])
                        positions=np.stack([np.interp(grid[valid],times,points[:,j]) for j in range(3)],-1)
                        centers=rotation(ego['rotation']).inv().apply(positions-ego['translation'])[:,:2]
                        rots=Rotation.concatenate([rotation(a['rotation']) for a in track])
                        relative=rotation(ego['rotation']).inv()*Slerp(times,rots)(grid[valid])
                        yaw=relative.as_euler('xyz')[:,2]
                        dimensions=np.stack([np.interp(grid[valid],times,sizes[:,j]) for j in range(3)],-1)
                        clearance=signed_point_box_distance(future[valid],centers,yaw,dimensions)
                        record.update(min_logged_centerline_clearance_m=float(clearance.min()),
                            future_grid_indices=np.flatnonzero(valid).tolist(),agent_centers_xy=centers.tolist(),
                            agent_yaws=yaw.tolist(),agent_sizes=dimensions.tolist())
            results.append(record)
        print(json.dumps({'scene':scene,'projected_boxes':len(projected)}),flush=True)
    summary={'n_pairs':len(results),'n_scenes':len(selected),'automatic_3d_candidates':sum(r['status']=='automatic_3d_candidate' for r in results),
        'full_future_candidates':sum(r.get('valid_future_steps',0)==64 for r in results),
        'limitations':['Automatic projection matches are not verified edit identities',
          'Traffic lights lack these nuScenes 3D object annotations',
          'Future tracks are logged, nonreactive, and not extrapolated',
          'Clearance is ego-centerline distance, excluding ego footprint and road rules',
          'No counterfactual optimal actions or physical safety labels are generated']}
    (args.out/'records.json').write_text(json.dumps(results,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest={'selection':'All records in provided detection/pair list; no model-response selection',
        'pairs_path':str(args.pairs),'pairs_sha256':hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        'matching':{'minimum_iou':.5,'minimum_iou_margin':.15},
        'table_stats':{str(tables/n):{'bytes':(tables/n).stat().st_size,'mtime_ns':(tables/n).stat().st_mtime_ns} for n in
            ['sample.json','sample_data.json','ego_pose.json','sample_annotation.json','instance.json','calibrated_sensor.json']}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
