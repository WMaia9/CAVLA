"""Prospectively landmarked vehicle transitions in full local nuScenes annotations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.nuscenes_geometry import iter_array,rotation,projected_box


def first_landmark(times,positions):
    for i in range(6,len(times)):
        gaps=np.diff(times[i-6:i+1])*1e-6
        if np.any((gaps<.25)|(gaps>.75)):continue
        speed=np.linalg.norm(np.diff(positions[i-6:i+1],axis=0),axis=1)/gaps
        if speed[-2:].max()<=.5:return i
    return None


def future_event(times,positions,index,horizon=6.4):
    """Detection time for two fast intervals and >2m displacement; no extrapolation."""
    last=index
    for j in range(index+1,len(times)):
        gap=(times[j]-times[j-1])*1e-6
        if not .25<=gap<=.75:break
        last=j;elapsed=(times[j]-times[index])*1e-6
        if elapsed>horizon:break
        if j>=index+2:
            gaps=np.diff(times[j-2:j+1])*1e-6
            speed=np.linalg.norm(np.diff(positions[j-2:j+1],axis=0),axis=1)/gaps
            if speed.min()>1 and np.linalg.norm(positions[j]-positions[index])>2:
                return elapsed,min(horizon,(times[last]-times[index])*1e-6)
    return None,min(horizon,(times[last]-times[index])*1e-6)


def history_features(times,positions,yaw):
    t=(times-times[-1])*1e-6;gaps=np.diff(t)
    xy=positions[:,:2]-positions[-1,:2]
    c,s=np.cos(yaw),np.sin(yaw);xy=xy@np.array([[c,-s],[s,c]])
    v=np.diff(xy,axis=0)/gaps[:,None];speed=np.linalg.norm(v,axis=1)
    acceleration=(v[-1]-v[-2])/((gaps[-1]+gaps[-2])/2)
    base=np.r_[speed[-2:],speed.mean(),speed.max(),acceleration,np.linalg.norm(xy[0]),-t[0]]
    design=np.c_[np.ones(len(t)),t];residual=xy-design@np.linalg.lstsq(design,xy,rcond=None)[0]
    lag=lambda a:float(np.sum(a[:-1]*a[1:])/np.sqrt(max(np.sum(a[:-1]**2)*np.sum(a[1:]**2),1e-20)))
    extra=np.r_[np.var(v,axis=0),np.var(residual,axis=0),lag(residual),lag(v-v.mean(0)),
        np.log((np.var(v[3:])+1e-6)/(np.var(v[:3])+1e-6)),np.mean(speed[3:])-np.mean(speed[:3])]
    raw=np.r_[xy[:-1].reshape(-1),gaps]
    return base,extra,raw,bool(speed.max()<=.5)


def summarize(rows):
    result={}
    for group,rr in [('all_landmarks',rows),('strict_stationary',[r for r in rows if r['strict_stationary']]),
                     ('counter_frames_all_projected',[r for r in rows if r.get('cohort')=='counter_frame'])]:
        result[group]={'agents':len(rr),'logs':len({r['log'] for r in rr}),'scenes':len({r['scene'] for r in rr}),'horizons':{}}
        for horizon in [2.,4.,6.4]:
            event=[r for r in rr if r['event_seconds'] is not None and r['event_seconds']<=horizon]
            negative=[r for r in rr if r['followup_seconds']>=horizon and (r['event_seconds'] is None or r['event_seconds']>horizon)]
            result[group]['horizons'][str(horizon)]={'events':len(event),'event_logs':len({r['log'] for r in event}),
                'known_negatives':len(negative),'censored_before_horizon':len(rr)-len(event)-len(negative)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path('/mnt/datasets/nuScenes');tables=root/'v1.0-trainval'
    samples={r['token']:r for r in json.loads((tables/'sample.json').read_text())}
    scenes={r['token']:r for r in json.loads((tables/'scene.json').read_text())}
    categories={r['token']:r['name'] for r in json.loads((tables/'category.json').read_text())}
    instances={r['token']:categories[r['category_token']] for r in json.loads((tables/'instance.json').read_text())}
    attrs={r['token']:r['name'] for r in json.loads((tables/'attribute.json').read_text())}
    allowed={k:v for k,v in instances.items() if v.startswith(('vehicle.car','vehicle.truck','vehicle.bus','vehicle.motorcycle','vehicle.bicycle'))}
    print('Streaming vehicle annotations',flush=True);tracks={}
    for a in iter_array(tables/'sample_annotation.json'):
        if a['instance_token'] in allowed:
            tracks.setdefault(a['instance_token'],[]).append({k:a[k] for k in ['token','sample_token','translation','rotation','size','attribute_tokens','visibility_token']})
    print('Tracks',len(tracks),flush=True)
    ranked={r['scene_name']:r for r in csv.DictReader(Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/scenes_ranked.csv').open())}
    wanted_sd={r['sd_token'] for r in ranked.values()};sd={r['token']:r for r in iter_array(tables/'sample_data.json') if r['token'] in wanted_sd}
    wanted_ego={r['ego_pose_token'] for r in sd.values()};egos={r['token']:r for r in iter_array(tables/'ego_pose.json') if r['token'] in wanted_ego}
    sensors={r['token']:r for r in json.loads((tables/'calibrated_sensor.json').read_text())}
    rows=[];counter=[];arrays={k:[] for k in ['basic','fluctuation','raw_history']}
    for instance,track in sorted(tracks.items()):
        track.sort(key=lambda a:samples[a['sample_token']]['timestamp'])
        times=np.array([samples[a['sample_token']]['timestamp'] for a in track],np.int64);pos=np.array([a['translation'] for a in track])
        scene=scenes[samples[track[0]['sample_token']]['scene_token']];i=first_landmark(times,pos)
        indices=[]
        if i is not None:indices.append(('first_landmark',i,None))
        if scene['name'] in ranked:
            meta=ranked[scene['name']];camera=sd[meta['sd_token']];at=np.searchsorted(times,camera['timestamp'],side='right')-1
            if at>=6 and (camera['timestamp']-times[at])*1e-6<=.75:
                past_t=times[at-6:at+1];gaps=np.diff(past_t)*1e-6
                if np.all((gaps>=.25)&(gaps<=.75)):
                    speed=np.linalg.norm(np.diff(pos[at-2:at+1],axis=0),axis=1)/(np.diff(times[at-2:at+1])*1e-6)
                    if speed.max()<=.5:
                        box=projected_box(track[at],egos[camera['ego_pose_token']],sensors[camera['calibrated_sensor_token']])
                        if box is not None and np.prod(np.subtract(box[2:],box[:2]))>=256:
                            indices.append(('counter_frame',at,box))
        for cohort,index,box in indices:
            ann=track[index];event,followup=future_event(times,pos,index)
            yaw=rotation(ann['rotation']).as_euler('xyz')[2]
            basic,extra,raw,strict=history_features(times[index-6:index+1],pos[index-6:index+1],yaw)
            names=[attrs[t] for t in ann['attribute_tokens']]
            row=dict(cohort=cohort,instance=instance,annotation=ann['token'],sample=ann['sample_token'],scene=scene['name'],log=scene['log_token'],
                timestamp=int(times[index]),category=allowed[instance],size=ann['size'],attribute=names[0] if len(names)==1 else 'unknown',
                visibility=ann['visibility_token'],event_seconds=event,followup_seconds=followup,strict_stationary=strict,projected_box=box)
            if cohort=='counter_frame':
                camera=sd[ranked[scene['name']]['sd_token']]
                row.update(input_camera_timestamp=camera['timestamp'],image=str(Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')/scene['name']/(scene['name']+'_front.jpg')),
                    annotation_age_seconds=(camera['timestamp']-times[index])*1e-6)
                counter.append(row)
            else:
                rows.append(row);arrays['basic'].append(basic);arrays['fluctuation'].append(extra);arrays['raw_history'].append(raw)
    summary={'principal':summarize(rows),'counter_frames':summarize(counter),'vehicle_tracks':len(tracks)}
    np.savez_compressed(args.out/'features.npz',**{k:np.array(v) for k,v in arrays.items()})
    (args.out/'landmarks.json').write_text(json.dumps(rows,indent=2));(args.out/'counter_frame_objects.json').write_text(json.dumps(counter,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'manifest.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(Path('docs/PROTOCOLO_TRANSICOES_E_OBSERVABILIDADE_V1.md').read_bytes()).hexdigest(),
        cohort='One first past-eligible landmark per vehicle instance, all850scenes; second inventory all vehicles projectable in ranked frames',
        caveats=['Only principal cohort prospective landmark timestamps; Counter-frame labels currently relative to past annotation timestamp, not camera',
          'Counter projection no occlusion verification; no extrapolation to camera timestamp','Events are operational annotation transitions, not mental intent',
          'All scenes/logs potentially examined before; no independent holdout','Principal inventory not restricted to camera field of view']),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
