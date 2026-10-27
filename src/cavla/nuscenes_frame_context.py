"""Build a documented nuScenes multiview/video adaptation, not native sensors.

nuScenes has no matching front-tele camera. The fourth slot explicitly duplicates
front images; FOVs also differ from Physical AI AV. Never label this native input.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cavla.nuscenes_geometry import iter_array


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--keyframes-only',action='store_true')
    p.add_argument('--front-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    pairs=json.loads(args.pairs.read_text());selected={r['scene'] for r in pairs}
    nusc=Path('/mnt/datasets/nuScenes');root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen')
    ranked={r['scene_name']:r for r in csv.DictReader((root/'scenes_ranked.csv').open()) if r['scene_name'] in selected}
    samples={r['token']:r for r in json.loads((nusc/'v1.0-trainval/sample.json').read_text())}
    scenes={samples[r['sample_token']]['scene_token'] for r in ranked.values()}
    sample_tokens={k for k,v in samples.items() if v['scene_token'] in scenes}
    cameras=['CAM_FRONT'] if args.front_only else ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']
    records={r['token']:r for r in iter_array(nusc/'v1.0-trainval/sample_data.json')
             if r['sample_token'] in sample_tokens and any('/'+c+'/' in r['filename'] for c in cameras)}
    by_sample={}
    for r in records.values():
        if r['is_key_frame']:
            camera=Path(r['filename']).parent.name
            by_sample[r['sample_token'],camera]=r
    manifest={'protocol':'Camera-major 4x4: front-left, front, front-right, duplicate-front proxy',
              'limitations':['Not native sensor layout: missing front-tele view is duplicated front',
                'nuScenes camera FOVs differ from the model training sensors',
                'Latest front edit is applied to both front-derived slots; other views/history remain recorded',
                'Pixel intervention control, not a physically consistent multi-view object removal'],
              'latest_front_slots':[[1,3],[3,3]],'scenes':{}}
    spacing=.5 if args.keyframes_only else .1
    manifest['time_spacing_seconds']=spacing
    if args.keyframes_only:
        manifest['limitations'].append('Only local 2Hz keyframes are available; temporal spacing differs from the 10Hz native example')
    if args.front_only:
        manifest['protocol']='Front-only temporal context, four real past frames, no zero padding or duplicated views'
        manifest['latest_front_slots']=[[0,3]]
        manifest['limitations']=['Reduced camera subset, not native multiview input',
            'nuScenes sensor/FOV differs from the model training data',
            'Only latest frame is edited; recorded past is unchanged',
            'Pixel intervention, not physically consistent object deletion throughout video']
    for scene in sorted(selected):
        meta=ranked[scene];t0=int(meta['timestamp_us']);times=t0+np.array([-3,-2,-1,0])*spacing*1e6
        selected_frames=[];arrays=[]
        for camera in cameras:
            current=by_sample[meta['sample_token'],camera];track=[]
            while True:
                track.append(current)
                if (current['timestamp']<=times.min() and (not args.keyframes_only or current['is_key_frame'])) or current['prev'] not in records:break
                current=records[current['prev']]
            frames=[]
            for target in times:
                available=[r for r in track if r['timestamp']<=target and (not args.keyframes_only or r['is_key_frame'])]
                if not available:raise RuntimeError(f'No past-only frame for {scene} {camera} {target}')
                chosen=max(available,key=lambda r:r['timestamp'])
                age=(target-chosen['timestamp'])*1e-6
                if age>(.62 if args.keyframes_only else .12):raise RuntimeError(f'Stale frame {age}s for {scene} {camera}')
                path=nusc/chosen['filename']
                with Image.open(path) as im:array=np.array(im.convert('RGB').resize((1920,1080),Image.Resampling.LANCZOS)).transpose(2,0,1)
                frames.append(array)
                selected_frames.append({'camera':camera,'target_timestamp':int(target),'timestamp':chosen['timestamp'],
                    'age_seconds':age,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            arrays.append(np.stack(frames))
        if not args.front_only:arrays.append(arrays[1].copy())
        frames=np.stack(arrays)
        np.savez_compressed(args.out/f'{scene}_frames.npz',frames=frames)
        manifest['scenes'][scene]={'reference_front_sd_token':meta['sd_token'],'t0':t0,'frames':selected_frames,
            'npz_sha256':hashlib.sha256((args.out/f'{scene}_frames.npz').read_bytes()).hexdigest()}
        print(json.dumps({'scene':scene,'shape':list(frames.shape),'max_age_seconds':max(r['age_seconds'] for r in selected_frames)}),flush=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
