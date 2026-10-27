"""Post-hoc decomposition and influence checks for held-out hazard predictions."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from cavla.transition_clock_probe import EDGES,path_loss
from cavla.transition_history_probe import log_mean


def bootstrap_difference(delta,logs):
    unique=sorted(set(logs));values=np.array([delta[logs==l].mean() for l in unique])
    rng=np.random.default_rng(20261021);draws=rng.integers(len(unique),size=(10000,len(unique)))
    return {'logs':len(unique),'difference':float(values.mean()),'conditional_log_bootstrap_95':np.quantile(values[draws].mean(1),[.025,.975]).tolist(),
        'fraction_logs_improved':float(np.mean(values<0)),'leave_one_log_out_range':[(float(values.sum()-v)/(len(values)-1)) for v in [values.max(),values.min()]] if len(values)>1 else None}


def same_group_auc(rows,ids,event,follow,risk,grouping,horizon):
    groups=defaultdict(list)
    for j,i in enumerate(ids):
        y=np.isfinite(event[j]) and event[j]<=horizon
        if y or follow[j]>=horizon:groups[rows[i][grouping]].append((j,int(y)))
    scores=[];logs=[];pairs=0;agents=0;events=0
    for group in groups.values():
        pos=[j for j,y in group if y];neg=[j for j,y in group if not y]
        if not pos or not neg:continue
        d=risk[pos,None]-risk[np.array(neg)[None,:]]
        scores.append(float(np.mean((d>0)+.5*(d==0))));logs.append(rows[ids[group[0][0]]]['log']);pairs+=d.size;agents+=len(group);events+=len(pos)
    if not scores:return {'mixed_groups':0}
    return {'mixed_groups':len(scores),'agents':agents,'events':events,'pairs':pairs,'logs':len(set(logs)),
        'log_mean_group_auc':log_mean(np.array(scores),np.array(logs))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=json.loads(Path('artifacts/front_transition_data_v1_20261021/objects.json').read_text());output={}
    with np.load(args.predictions/'predictions.npz') as z:arrays={k:z[k] for k in z.files}
    for cohort in ['principal','strict']:
        ids=arrays[cohort+'_ids'];logs=arrays[cohort+'_logs'];event=np.array([np.nan if rows[i]['event_seconds'] is None else rows[i]['event_seconds'] for i in ids]);follow=np.array([rows[i]['followup_seconds'] for i in ids])
        observed=np.isfinite(event);eids=np.flatnonzero(observed);bins=np.searchsorted(EDGES[1:],event[eids],side='left')
        methods={key[len(cohort)+1:]:value for key,value in arrays.items() if key.startswith(cohort+'_') and key not in [cohort+'_ids',cohort+'_logs']}
        results={};timing={};losses={}
        for name,hz in methods.items():
            hz=np.clip(hz,1e-8,1-1e-8);surv=np.cumprod(1-hz,axis=1);pmf=hz*np.c_[np.ones(len(hz)),surv[:,:-1]]
            timing[name]=-np.log((pmf[eids,bins]/(1-surv[eids,-1])).clip(min=1e-12))
            losses[name]=path_loss(event,follow,hz)
            result={'timing_nll_given_observed_event':log_mean(timing[name],logs[eids]),'mixed_groups':{}}
            for h in [4.,6.4]:
                b=int(np.searchsorted(EDGES,h))-1;risk=1-surv[:,b]
                for grouping in ['image','scene']:result['mixed_groups'][grouping+'_'+str(h)]=same_group_auc(rows,ids,event,follow,risk,grouping,h)
            results[name]=result
        comparisons={}
        pairs=[(name,baseline) for name in methods if name.startswith(('raw_clip','raw_r1','raw_qwen','raw_temporal','raw_repeat')) for baseline in ['raw','raw_rff','bin_hazard']]
        for suffix in ['roi','both']:
            if 'raw_temporal_'+suffix in methods and 'raw_repeat_'+suffix in methods:
                pairs.extend([('raw_temporal_'+suffix,'raw_repeat_'+suffix),('raw_temporal_'+suffix,'raw_qwen_both'),('raw_temporal_'+suffix,'raw_clip_both')])
        for name,baseline in pairs:
            comparisons[name+'_minus_'+baseline]={'timing':bootstrap_difference(timing[name]-timing[baseline],logs[eids]),
                'whole_observed_path':bootstrap_difference(losses[name]-losses[baseline],logs)}
        output[cohort]={'methods':results,'comparisons':comparisons}
    (args.out/'summary.json').write_text(json.dumps(output,indent=2));(args.out/'manifest.json').write_text(json.dumps({'predictions':str(args.predictions),
        'scope':'Post-hoc diagnostics; mixed groups and event timing are selected by future outcome; no independent confirmation.'},indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
