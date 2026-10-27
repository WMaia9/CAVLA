"""Past-only frontal vehicle landmarks, using all available local keyframes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.nuscenes_geometry import iter_array,rotation,projected_box
from cavla.motion_transition_inventory import history_features,future_event,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    root=Path('/mnt/datasets/nuScenes');tables=root/'v1.0-trainval'
    samples={r['token']:r for r in json.loads((tables/'sample.json').read_text())}
    scenes={r['token']:r for r in json.loads((tables/'scene.json').read_text())}
    categories={r['token']:r['name'] for r in json.loads((tables/'category.json').read_text())}
    instances={r['token']:categories[r['category_token']] for r in json.loads((tables/'instance.json').read_text())}
    allowed={k:v for k,v in instances.items() if v.startswith(('vehicle.car','vehicle.truck','vehicle.bus','vehicle.motorcycle','vehicle.bicycle'))}
    attrs={r['token']:r['name'] for r in json.loads((tables/'attribute.json').read_text())}
    sensors={r['token']:r for r in json.loads((tables/'sensor.json').read_text())}
    calibrations={r['token']:r for r in json.loads((tables/'calibrated_sensor.json').read_text())}
    front_calibrations={k for k,c in calibrations.items() if sensors[c['sensor_token']]['channel']=='CAM_FRONT'}
    print('Scanning front keyframes and poses',flush=True)
    front={r['sample_token']:r for r in iter_array(tables/'sample_data.json') if r['is_key_frame'] and r['calibrated_sensor_token'] in front_calibrations and (root/r['filename']).is_file()}
    wanted_ego={r['ego_pose_token'] for r in front.values()};egos={r['token']:r for r in iter_array(tables/'ego_pose.json') if r['token'] in wanted_ego}
    print('Scanning annotated vehicle tracks',flush=True);tracks={}
    for a in iter_array(tables/'sample_annotation.json'):
        if a['instance_token'] in allowed:tracks.setdefault(a['instance_token'],[]).append({k:a[k] for k in ['token','sample_token','translation','rotation','size','attribute_tokens','visibility_token']})
    rows=[];arrays={k:[] for k in ['basic','fluctuation','raw_history','relative_geometry']}
    for instance,track in sorted(tracks.items()):
        track.sort(key=lambda a:samples[a['sample_token']]['timestamp'])
        times=np.array([samples[a['sample_token']]['timestamp'] for a in track],np.int64);pos=np.array([a['translation'] for a in track])
        scene=scenes[samples[track[0]['sample_token']]['scene_token']]
        for current in range(6,len(track)):
            camera=front.get(track[current]['sample_token'])
            if camera is None:continue
            index=int(np.searchsorted(times,camera['timestamp'],side='right')-1)
            if index<6:continue
            age=(camera['timestamp']-times[index])*1e-6
            if not 0<=age<=.75:continue
            ann=track[index];gaps=np.diff(times[index-6:index+1])*1e-6
            if np.any((gaps<.25)|(gaps>.75)) or int(ann['visibility_token'])<3:continue
            velocity=(pos[index]-pos[index-1])/gaps[-1]
            speed=np.linalg.norm(np.diff(pos[index-2:index+1],axis=0),axis=1)/gaps[-2:]
            if speed.max()>.5:continue
            adjusted=dict(ann);adjusted['translation']=(pos[index]+age*velocity).tolist()
            ego=egos[camera['ego_pose_token']];calibration=calibrations[camera['calibrated_sensor_token']]
            box=projected_box(adjusted,ego,calibration)
            if box is None:continue
            wh=np.subtract(box[2:],box[:2])
            if wh.min()<16 or wh.prod()<1024:continue
            event,follow=future_event(times,pos,index,6.4+age)
            event=None if event is None else event-age;follow=max(0.,min(6.4,follow-age))
            if event is not None:assert 0<event<=6.4+1e-8
            yaw=rotation(ann['rotation']).as_euler('xyz')[2]
            basic,fluctuation,raw,strict=history_features(times[index-6:index+1],pos[index-6:index+1],yaw)
            inv=rotation(ego['rotation']).inv();center=inv.apply(np.array(adjusted['translation'])-ego['translation']);vel=inv.apply(velocity)
            relative_yaw=(inv*rotation(ann['rotation'])).as_euler('xyz')[2]
            geometry=np.r_[center[:2],vel[:2],np.cos(relative_yaw),np.sin(relative_yaw),np.array(box)/[1600,900,1600,900],age]
            labels=[attrs[t] for t in ann['attribute_tokens']]
            row=dict(instance=instance,annotation=ann['token'],sample=ann['sample_token'],camera_sample=camera['sample_token'],scene=scene['name'],log=scene['log_token'],
                timestamp=int(camera['timestamp']),past_annotation_timestamp=int(times[index]),annotation_age_seconds=age,
                category=allowed[instance],size=ann['size'],attribute=labels[0] if len(labels)==1 else 'unknown',visibility=ann['visibility_token'],
                event_seconds=event,followup_seconds=follow,strict_stationary=strict,projected_box=box,image=str(root/camera['filename']),
                camera_token=camera['token'])
            rows.append(row)
            for key,value in zip(arrays,[basic,fluctuation,raw,geometry]):arrays[key].append(value)
            break
    summary=summarize(rows)
    np.savez_compressed(args.out/'features.npz',**{k:np.array(v) for k,v in arrays.items()})
    (args.out/'objects.json').write_text(json.dumps(rows,indent=2));(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'manifest.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(Path('docs/PROTOCOLO_TRANSICOES_E_OBSERVABILIDADE_V1.md').read_bytes()).hexdigest(),
        selection='First past-eligible frontal camera per instance; visibility>=3,box>=1024px²,minside16; no future-selection',
        front_keyframes_available=len(front),counterfactual_actions=False,limitations=['Projectability and past visibility not perfect image visibility','Annotation speed includes measurement error','Human attributes not prospective intent labels','No new untouched evaluation logs']),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
