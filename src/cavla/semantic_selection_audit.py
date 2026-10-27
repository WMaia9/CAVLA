"""Audit semantic cohort expansion without requiring a complete logged future.

No new model evaluation. Uses all existing automatic 2D/3D associations; counts
semantic states and mixed scenes before versus after the old future filter.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cavla.nuscenes_geometry import iter_array


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sources=[('development_pool','nuscenes_geometry_v3_20261021'),('previously_used_holdout_pool','composition_holdout_geometry_v1_20261021')]
    candidates={}
    for source,folder in sources:
        for r in json.loads((Path('artifacts')/folder/'records.json').read_text()):
            if r.get('status')!='automatic_3d_candidate':continue
            key=r['scene'],r['instance_token']
            rank=(r['best_iou'],r['best_iou']-r['runner_up_iou'],r['stem'])
            if key not in candidates or rank>candidates[key][0]:candidates[key]=(rank,source,r)
    tables=Path('/mnt/datasets/nuScenes/v1.0-trainval')
    times={s['token']:s['timestamp'] for s in json.loads((tables/'sample.json').read_text())}
    names={a['token']:a['name'] for a in json.loads((tables/'attribute.json').read_text())}
    ranked={r['scene_name']:int(r['timestamp_us']) for r in csv.DictReader(Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/scenes_ranked.csv').open())}
    wanted={r['instance_token'] for _,_,r in candidates.values()};tracks={}
    for a in iter_array(tables/'sample_annotation.json'):
        if a['instance_token'] in wanted:tracks.setdefault(a['instance_token'],[]).append(a)
    rows=[]
    for _,source,r in candidates.values():
        timestamp=ranked[r['scene']]
        past=sorted([(times[a['sample_token']],a) for a in tracks[r['instance_token']] if times[a['sample_token']]<=timestamp],key=lambda z:z[0])
        if len(past)<2:continue
        (t0,a0),(t1,a1)=past[-2:];gap=(t1-t0)*1e-6;age=(timestamp-t1)*1e-6
        if not 0<gap<=.75 or age>.75:continue
        # 3D world speed is used only for this data-inventory gate, not a change
        # to the previously trained ego-XY-speed cohort.
        speed=float(np.linalg.norm(np.array(a1['translation'])-a0['translation'])/gap)
        labels=[names[t] for t in a1['attribute_tokens']]
        state=labels[0] if len(labels)==1 else 'unknown'
        if state not in ['vehicle.parked','vehicle.stopped'] or speed>.5:continue
        rows.append({'source':source,'scene':r['scene'],'stem':r['stem'],'instance_token':r['instance_token'],
            'past_annotation_token':a1['token'],'past_timestamp':t1,'input_timestamp':timestamp,'past_speed_3d_mps':speed,
            'state':state,'valid_future_steps':r.get('valid_future_steps',0),'old_full_future_eligible':r.get('valid_future_steps',0)==64,
            'best_iou':r['best_iou'],'iou_margin':r['best_iou']-r['runner_up_iou']})
    summary={}
    for source,_ in sources:
        summary[source]={}
        for tag,subset in [('without_future_filter',[r for r in rows if r['source']==source]),
                           ('with_full_future_filter',[r for r in rows if r['source']==source and r['old_full_future_eligible']])]:
            scenes=sorted({r['scene'] for r in subset})
            mixed=[s for s in scenes if len({r['state'] for r in subset if r['scene']==s})==2]
            summary[source][tag]={'agents':len(subset),'scenes':len(scenes),'mixed_scenes':len(mixed),
                'mixed_scene_agents':sum(r['scene'] in mixed for r in subset),
                'parked':sum(r['state']=='vehicle.parked' for r in subset),'stopped':sum(r['state']=='vehicle.stopped' for r in subset)}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'objects.json').write_text(json.dumps(rows,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':'Existing automatic-match pool; no new model comparison or untouched holdout',
        'selection':'IoU/margin-qualified, deduplicate by scene-instance best match, two recent past annotations,3D speed<=.5,parked/stopped; no model-response selection',
        'limitations':['Counts are for existing automatic detections, not all road agents','World3D speed differs slightly from prior egoXY speed','No future-completeness filter needed for current state labels','No reactive or counterfactual action labels']},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
