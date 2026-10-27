"""Separate administrative scene end from annotation loss in the frontal cohort.

Descriptive follow-up audit only: does not identify why an actor stopped being annotated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.nuscenes_geometry import iter_array


def observation_window(times, timestamp, horizon=6.4):
    index=int(np.searchsorted(times,timestamp,side='right')-1)
    if index<0:raise ValueError('No preceding annotation')
    last=index;reason='track_end'
    for j in range(index+1,len(times)):
        gap=(times[j]-times[j-1])*1e-6
        if not .25<=gap<=.75:
            reason='annotation_gap';break
        last=j
        if (times[j]-timestamp)*1e-6>=horizon:
            reason='horizon_complete';break
    return max(0.,min(horizon,(times[last]-timestamp)*1e-6)),reason


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data=Path('artifacts/front_transition_data_v1_20261021');rows=json.loads((data/'objects.json').read_text())
    tables=Path('/mnt/datasets/nuScenes/v1.0-trainval')
    samples={r['token']:r for r in json.loads((tables/'sample.json').read_text())}
    ends={}
    for s in samples.values():ends[s['scene_token']]=max(ends.get(s['scene_token'],0),s['timestamp'])
    wanted={r['instance'] for r in rows};tracks={k:[] for k in wanted}
    for a in iter_array(tables/'sample_annotation.json'):
        if a['instance_token'] in tracks:
            tracks[a['instance_token']].append((samples[a['sample_token']]['timestamp'],a['translation']))
    output=[];cache={}
    for i,r in enumerate(rows):
        track=sorted(tracks[r['instance']]);times=np.array([t for t,_ in track],np.int64)
        positions=np.array([p for _,p in track]);available,reason=observation_window(times,r['timestamp'])
        scene_end=ends[samples[r['sample']]['scene_token']]
        administrative=max(0.,(scene_end-r['timestamp'])*1e-6)
        if reason=='track_end' and (scene_end-times[-1])*1e-6<=.75:reason='near_administrative_end'
        event=r['event_seconds']
        output.append(dict(index=i,instance=r['instance'],log=r['log'],strict=r['strict_stationary'],attribute=r['attribute'],
            observation_seconds=available,reason=reason,scene_seconds_remaining=administrative,
            event_seconds=event,original_followup_seconds=r['followup_seconds']))
        cache[str(i)+'_times']=times;cache[str(i)+'_positions']=positions
        # Existing event code deliberately ends follow-up at an event; do not count that as censoring.
        if event is None:assert abs(available-r['followup_seconds'])<1e-6
    summaries={}
    for strict in [False,True]:
        rr=[r for r in output if not strict or r['strict']]
        summary={'agents':len(rr),'horizons':{}}
        for h in [2.,4.,6.4]:
            censored=[r for r in rr if r['observation_seconds']<h and (r['event_seconds'] is None or r['event_seconds']>h)]
            counts={reason:sum(r['reason']==reason for r in censored) for reason in sorted({r['reason'] for r in output})}
            summary['horizons'][str(h)]={'censored_unknown':len(censored),'reasons':counts,
                'administrative_horizon_unavailable':sum(r['scene_seconds_remaining']<h for r in censored)}
        summaries['strict' if strict else 'principal']=summary
    np.savez_compressed(args.out/'tracks.npz',**cache)
    (args.out/'observations.json').write_text(json.dumps(output,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'objects_sha256':hashlib.sha256((data/'objects.json').read_bytes()).hexdigest(),
        'scope':'Descriptive; annotation ending before scene end does not prove occlusion or informative censoring.',
        'near_end_tolerance_seconds':.75},indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
