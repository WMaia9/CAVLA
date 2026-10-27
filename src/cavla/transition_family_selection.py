"""Select predictor families strictly from the inner validation records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.transition_clock_probe import EDGES,path_loss
from cavla.transition_history_probe import log_mean,metrics
from cavla.transition_clock_diagnostics import bootstrap_difference


def validation_selection(predictions,folds,validation,choices):
    result=np.full_like(next(iter(predictions.values())),np.nan);records=[]
    for f in range(5):
        winner=min(choices,key=lambda name:validation[f,name]);use=folds==f
        result[use]=predictions[winner][use];records.append({'fold':f,'winner':winner,'validation_nll':validation[f,winner]})
    assert np.isfinite(result).all()
    return result,records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--tree',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    pm=json.loads((args.probe/'manifest.json').read_text());tm=json.loads((args.tree/'manifest.json').read_text())
    assert pm['objects_sha256']==tm['objects_sha256'];assert Path(pm['data']).resolve()==Path(tm['data']).resolve()
    rows=json.loads((Path(pm['data'])/'objects.json').read_text());pr=json.loads((args.probe/'records.json').read_text());tr=json.loads((args.tree/'records.json').read_text())
    with np.load(args.probe/'predictions.npz') as z:pa={k:z[k] for k in z.files}
    with np.load(args.tree/'predictions.npz') as z:ta={k:z[k] for k in z.files}
    summary={};save={};records=[]
    for cohort in ['principal','strict']:
        ids=pa[cohort+'_ids'];logs=pa[cohort+'_logs'];assert np.array_equal(ids,ta[cohort+'_ids']) and np.array_equal(logs,ta[cohort+'_logs'])
        event=np.array([np.nan if rows[i]['event_seconds'] is None else rows[i]['event_seconds'] for i in ids]);follow=np.array([rows[i]['followup_seconds'] for i in ids])
        folds=np.array([int(hashlib.sha256(l.encode()).hexdigest(),16)%5 for l in logs])
        predictions={key[len(cohort)+1:]:value for key,value in pa.items() if key.startswith(cohort+'_') and key not in [cohort+'_ids',cohort+'_logs']}
        predictions['numeric_tree']=ta[cohort+'_numeric_tree']
        validation={(r['fold'],r['method']):r['selection']['validation_nll'] for r in pr if r['cohort']==cohort}
        validation.update({(r['fold'],'numeric_tree'):r['selection']['validation_nll'] for r in tr if r['cohort']==cohort})
        numeric=['bin_hazard','basic','raw','raw_rff','numeric_tree'];families={'numeric':numeric}
        for prefix in ['clip','r1','qwen','temporal','repeat']:
            extra=[name for name in predictions if name.startswith('raw_'+prefix+'_')]
            if extra:families[prefix]=numeric+extra
        selected={};values={}
        for family,choices in families.items():
            hz,selection=validation_selection(predictions,folds,validation,choices);selected[family]=hz
            records.append({'cohort':cohort,'family':family,'choices':choices,'selections':selection})
            result={'observed_path_log_nll':log_mean(path_loss(event,follow,hz),logs),'horizons':{}}
            for h in [2.,4.,6.4]:
                y=np.isfinite(event)&(event<=h);known=y|(follow>=h);b=int(np.searchsorted(EDGES,h))-1;risk=1-np.prod(1-hz[:,:b+1],axis=1)
                result['horizons'][str(h)]=metrics(y[known].astype(float),risk[known],logs[known])
            values[family]=result
        comparisons={}
        pairs=[(name,'numeric') for name in selected if name!='numeric']
        if 'temporal' in selected and 'repeat' in selected:pairs.append(('temporal','repeat'))
        for name,base in pairs:comparisons[name+'_minus_'+base]=bootstrap_difference(path_loss(event,follow,selected[name])-path_loss(event,follow,selected[base]),logs)
        summary[cohort]={'methods':values,'comparisons':comparisons};save.update({cohort+'_ids':ids,cohort+'_logs':logs,**{cohort+'_'+k:v for k,v in selected.items()}})
    np.savez_compressed(args.out/'predictions.npz',**save);(args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'probe':str(args.probe),'tree':str(args.tree),'objects_sha256':pm['objects_sha256'],
        'scope':'Adaptive family-selection control, inner validation only; conditional bootstrap, no untouched confirmation'},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
