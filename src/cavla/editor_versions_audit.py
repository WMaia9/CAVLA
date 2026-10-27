"""Descriptive comparison of local raw/composited archival inference versions.

Exact agreement of original predictions does not prove all settings are matched.
Use the separately executed pixel-factorial experiment for controlled contrasts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    rows=[];originals=[]
    for directory in sorted((args.root/'comp_results_fin_comb').glob('scene-*')):
        raw_dir=args.root/'comp_results'/directory.name
        p=raw_dir/'trajectories/original_traj.npz';q=directory/'trajectories/original_traj.npz'
        if not p.exists() or not q.exists():continue
        with np.load(p) as z:orig0=z['pred_xyz']
        with np.load(q) as z:orig1=z['pred_xyz']
        if orig0.shape!=orig1.shape:continue
        error=float(np.abs(orig0-orig1).max())
        originals.append({'scene':directory.name,'original_maxabs':error})
        for clean_path in sorted((directory/'trajectories').glob('*_traj.npz')):
            raw_path=raw_dir/'trajectories'/clean_path.name
            if clean_path.name=='original_traj.npz' or not raw_path.exists():continue
            with np.load(raw_path) as z:x=z['pred_xyz']
            with np.load(clean_path) as z:y=z['pred_xyz']
            if x.shape!=y.shape:continue
            rows.append({'scene':directory.name,'stem':clean_path.stem.removesuffix('_traj'),
                'original_maxabs':error,
                'raw_response_m':float(np.linalg.norm(x[...,:2]-orig0[...,:2],axis=-1).mean()),
                'clean_response_m':float(np.linalg.norm(y[...,:2]-orig1[...,:2],axis=-1).mean()),
                'raw_clean_distance_m':float(np.linalg.norm(x[...,:2]-y[...,:2],axis=-1).mean())})
    summary={'n_scenes':len(originals),'n_pairs':len(rows),
        'originals_exact':sum(x['original_maxabs']==0 for x in originals),
        **{f'median_{key}':float(np.median([r[key] for r in rows])) for key in
           ['raw_response_m','clean_response_m','raw_clean_distance_m']},
        'limitation':'Versions not verified to differ only in editor processing; archival descriptive comparison, not artifact causal identification'}
    for name,value in [('records',rows),('original_controls',originals),('summary',summary)]:
        (args.out/f'{name}.json').write_text(json.dumps(value,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
