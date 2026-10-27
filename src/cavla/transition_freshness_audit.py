"""Diagnostic only: distinguish stale numerical state from pre-motion anticipation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=json.loads(Path('artifacts/front_transition_data_v1_20261021/objects.json').read_text());records=[]
    with np.load('artifacts/transition_observation_audit_v1_20261021/tracks.npz') as z:
        for i,r in enumerate(rows):
            times=z[str(i)+'_times'];pos=z[str(i)+'_positions'];j=int(np.searchsorted(times,r['timestamp'],side='right')-1)
            assert int(times[j])==r['past_annotation_timestamp']
            speed=None;before=None;next_delay=None
            if j+1<len(times):
                gap=(times[j+1]-times[j])*1e-6
                if .25<=gap<=.75:
                    speed=float(np.linalg.norm(pos[j+1]-pos[j])/gap);before=r['annotation_age_seconds']/gap;next_delay=(times[j+1]-r['timestamp'])*1e-6
            records.append({'index':i,'instance':r['instance'],'age':r['annotation_age_seconds'],'strict':r['strict_stationary'],
                'event':r['event_seconds'] is not None,'bracketing_interval_speed':speed,'interval_fraction_before_camera':before,'next_annotation_delay':next_delay})
    summary={}
    for name,rr in [('principal',records),('strict',[r for r in records if r['strict']]),('events',[r for r in records if r['event']]),('strict_events',[r for r in records if r['strict'] and r['event']])]:
        known=[r for r in rr if r['bracketing_interval_speed'] is not None];long=[r for r in known if r['age']>=.4]
        summary[name]={'agents':len(rr),'age_quantiles':np.quantile([r['age'] for r in rr],[0,.25,.5,.75,1]).tolist(),
            'fraction_age_ge_0_4':float(np.mean([r['age']>=.4 for r in rr])),'bracket_available':len(known),
            'bracket_speed_gt_0_5':sum(r['bracketing_interval_speed']>.5 for r in known),'bracket_speed_gt_1':sum(r['bracketing_interval_speed']>1 for r in known),
            'long_age_available':len(long),'long_age_bracket_speed_gt_0_5':sum(r['bracketing_interval_speed']>.5 for r in long),
            'long_age_next_annotation_delay_quantiles':np.quantile([r['next_annotation_delay'] for r in long],[0,.5,1]).tolist() if long else None}
    (args.out/'records.json').write_text(json.dumps(records,indent=2));(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':'Future annotation only for diagnostic speed over an interval spanning camera; never input or principal cohort selection',
        'limitations':['Average bracketing speed is not instantaneous camera-time speed','Does not locate physical onset within the interval','Observed transition definition unchanged']},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
