"""Within-scene state discrimination on existing out-of-log predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from cavla.semantic_state_probe import auc


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    with np.load(args.study/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    methods=[k.removeprefix('cv_') for k in pred if k.startswith('cv_')]
    details=[];summary={};boot={}
    for split,prefix in [('development','cv'),('previously_used_holdout','holdout')]:
        with np.load(Path('artifacts/semantic_attribute_data_v1_20261021')/f'{split}.npz') as z:
            use=z['slow_vehicle'];scenes=z['scenes'][use];logs=z['logs'][use];y=z['attributes'][use]=='vehicle.stopped'
        mixed=[s for s in sorted(set(scenes)) if len(set(y[scenes==s]))==2]
        scene_logs=np.array([logs[scenes==s][0] for s in mixed]);names=sorted(set(scene_logs))
        rng=np.random.default_rng(20261021);draws=rng.integers(len(names),size=(10000,len(names))) if len(names)>1 else None
        weights=np.zeros((10000,len(names))) if draws is not None else None
        if weights is not None:np.add.at(weights,(np.arange(len(draws))[:,None],draws),1)
        values={}
        for m in methods:
            predictions=pred[prefix+'_'+m]
            if predictions.ndim==1:predictions=predictions[None]
            scores=np.mean([[auc(y[scenes==s],pp[scenes==s]) for s in mixed] for pp in predictions],axis=0)
            values[m]=float(scores.mean())
            for s,score in zip(mixed,scores):details.append({'split':split,'scene':s,'log':str(logs[scenes==s][0]),'agents':int((scenes==s).sum()),'method':m,'auc':float(score)})
            if weights is not None:
                total=np.array([scores[scene_logs==log].sum() for log in names]);count=np.array([(scene_logs==log).sum() for log in names])
                boot[split,m]=(weights@total)/(weights@count)
        summary[split]={'mixed_scenes':len(mixed),'mixed_logs':len(names),'mixed_agents':int(np.isin(scenes,mixed).sum()),'scene_mean_auc':values,'comparisons':{}}
        if weights is not None:
            for family in ['logistic','rbf']:
                anchor=family+'_r1'
                if anchor not in values:continue
                for other in [m for m in methods if m.startswith(family+'_') and m!=anchor]:
                    delta=boot[split,anchor]-boot[split,other]
                    summary[split]['comparisons'][anchor+'_minus_'+other]={'auc_difference':values[anchor]-values[other],
                        'conditional_log_bootstrap_95':np.quantile(delta,[.025,.975]).tolist()}
        else:summary[split]['note']='One mixed log: no inferential bootstrap reported'
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(details,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
