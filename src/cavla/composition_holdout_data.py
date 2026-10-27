"""Prepare reserved-log query labels and verify source image provenance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cavla.geometric_learning import query_curve
from cavla.nuscenes_geometry import signed_point_box_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--reserved',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    reserved={s['name']:s for s in json.loads((args.reserved/'scenes.json').read_text())}
    source=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen');root=source/'nuscenes_per_scene'
    ranked={r['scene_name']:r for r in csv.DictReader((source/'scenes_ranked.csv').open())}
    rows=json.loads((args.geometry/'records.json').read_text());by_scene={}
    for r in rows:
        if r.get('status')=='automatic_3d_candidate' and r.get('valid_future_steps')==64:
            assert r['scene'] in reserved
            assert r['future_grid_indices']==list(range(64))
            by_scene.setdefault(r['scene'],[]).append(r)
    objects=[];selected=[];provenance=[];exclusions=[]
    for scene,records in sorted(by_scene.items()):
        folder=root/scene;image=folder/f'{scene}_front.jpg';original=Path('/mnt/datasets/nuScenes/samples/CAM_FRONT')/Path(ranked[scene]['image_path']).name
        if not original.exists():raise RuntimeError(f'Provenance source missing: {original}')
        with Image.open(image) as im:a=np.array(im.convert('RGB'))
        with Image.open(original) as im:b=np.array(im.convert('RGB'))
        if a.shape!=b.shape or a.shape[:2]!=(900,1600):raise RuntimeError(f'Image shape mismatch {scene}')
        error=float(np.abs(a.astype(float)-b.astype(float)).mean())
        if error>2:raise RuntimeError(f'Source image mismatch {scene}: {error}')
        with np.load(folder/f'{scene}_data.npz') as z:
            if not np.isfinite(z['ego_history_xyz']).all() or not np.isfinite(z['ego_history_rot']).all():
                raise RuntimeError(f'Invalid ego history {scene}')
        representatives={}
        for r in records:
            key=r['instance_token']
            if key not in representatives or (r['best_iou'],r['stem'])>(representatives[key]['best_iou'],representatives[key]['stem']):
                representatives[key]=r
        records=sorted(representatives.values(),key=lambda r:hashlib.sha256((scene+'/'+r['instance_token']).encode()).hexdigest())[:10]
        for r in records:
            selected.append(r);objects.append({'scene':scene,'stem':r['stem'],'fold':-1,
                'log_token':reserved[scene]['log_token'],'validation_group':reserved[scene]['log_token'],
                'instance_token':r['instance_token']})
        provenance.append({'scene':scene,'source_image':str(original),'mean_abs_decoded_rgb':error,
            'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
            'candidate_detections':len(representatives),'retained_agents':len(records)})
    for scene in reserved:
        if scene not in by_scene:exclusions.append({'scene':scene,'reason':'No available full-future matched detection; see geometry/input selection logs'})
    queries=[];modes=[]
    for mode,speeds,curves,horizons in [('id',[2,5,10,15],[-.03,-.015,0,.015,.03],[2,4,6.4]),
        ('ood',[3.5,7.5,12.5],[-.0225,-.0075,.0075,.0225],[3,5])]:
        for v in speeds:
            for k in curves:
                for h in horizons:queries.append([v/15,k/.03,h/6.4]);modes.append(mode)
    targets=[]
    for r in selected:
        centers=np.array(r['agent_centers_xy']);yaw=np.array(r['agent_yaws']);sizes=np.array(r['agent_sizes']);values=[]
        for v,k,h in queries:
            curve=query_curve(v*15,k*.03)
            distance=signed_point_box_distance(curve,centers,yaw,sizes)
            values.append(float(distance[:round(h*6.4*10)].min()<=2))
        targets.extend(values)
    np.savez_compressed(args.out/'dataset.npz',object_ids=np.repeat(np.arange(len(objects)),len(queries)),
        queries=np.tile(np.array(queries,np.float32),(len(objects),1)),query_modes=np.tile(np.array(modes),len(objects)),
        targets=np.array(targets,np.float32))
    (args.out/'objects.json').write_text(json.dumps(objects,indent=2));(args.out/'selected_geometry.json').write_text(json.dumps(selected,indent=2))
    (args.out/'empty_pairs.json').write_text('[]')
    manifest={'scope':'Reserved-log label construction only; no evaluation outcomes inspected',
        'objects':len(objects),'scenes':len(provenance),'logs':len({o['log_token'] for o in objects}),
        'geometry':str(args.geometry),'reserved':str(args.reserved),'provenance':provenance,'exclusions':exclusions,
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_HOLDOUT_LOGS_COMPOSICAO.md').read_bytes()).hexdigest(),
        'selection':'Highest-IoU representative per instance, up to ten agents by SHA256(scene/instance)',
        'limitations':['New subset selection is not restricted to scenes/objects with edited pairs','Automatic 2D/3D candidate joins remain unverified',
            'Same nuScenes collection; no guarantee about backbone pretraining exposure']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:manifest[k] for k in ['objects','scenes','logs']},indent=2))


if __name__=='__main__':main()
