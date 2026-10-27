"""Audit saved predictions, grouped splits, train-only PCA and model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cavla.geometric_learning import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((args.study/'manifest.json').read_text())['config']
    data=Path(config['data'])
    objects=json.loads((data/'objects.json').read_text())
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    rows=json.loads((args.study/'results.json').read_text())
    assert len(rows)==3*len(config['methods'])*len(config['seeds'])*2
    errors=[];pca_errors=[];groups=[]
    if config.get('validation'):
        for fold in range(3):
            split=json.loads((args.study/f'fold{fold}_split.json').read_text())
            parts=[split[k] for k in ['train_objects','validation_objects','test_objects']]
            assert sorted(sum(parts,[]))==list(range(len(objects)))
            key='log_token' if 'log_token' in objects[0] else 'scene'
            labels=[{objects[i][key] for i in part} for part in parts]
            for a,b in [(0,1),(0,2),(1,2)]:assert not labels[a]&labels[b]
            groups.append({'fold':fold,'unit':key,'train_validation_test_counts':[len(x) for x in labels]})
            with np.load(args.study/f'fold{fold}_original_pca.npz') as z:
                error=float(np.max(np.abs(z['mean']-d['originals'][parts[0]].mean(0))))
                assert error<1e-6;pca_errors.append(error)
            if config.get('spatial_features'):
                with np.load(config['spatial_features']) as z:x=z['features'].reshape(len(objects),-1)
                with np.load(args.study/f'fold{fold}_spatial_pca.npz') as z:
                    error=float(np.max(np.abs(z['mean']-x[parts[0]].mean(0))))
                    assert error<1e-6;pca_errors.append(error)
    for row in rows:
        stem=f"fold{row['fold']}_{row['method']}_seed{row['seed']}"
        with np.load(args.study/f'{stem}_predictions.npz') as z:i=z['indices'];p=z['probability']
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert np.all(d['object_folds'][d['object_ids'][i]]==row['fold'])
        select=d['query_modes'][i]==row['mode']
        measured=metrics(d['targets'][i][select],p[select])
        for key in ['auc','balanced_error','false_positive_rate','false_negative_rate']:
            error=abs(measured[key]-row[key]);assert error<1e-10;errors.append(error)
        if config.get('validation'):
            history=json.loads((args.study/f'{stem}_validation.json').read_text())
            best=min(history,key=lambda x:x['balanced_error'])
            assert row['selected_step']==best['step']
            assert row['validation_balanced_error']==best['balanced_error']
    result={'study':str(args.study),'models':len(rows)//2,'max_metric_error':max(errors),
        'max_train_only_pca_mean_error':max(pca_errors) if pca_errors else None,
        'groups':groups,'selection_verified':bool(config.get('validation')),
        'results_sha256':hashlib.sha256((args.study/'results.json').read_bytes()).hexdigest(),
        'scope':'Consistency audit, not independent replication or novelty validation'}
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
