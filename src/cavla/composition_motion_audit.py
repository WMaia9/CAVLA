"""Diagnostic effect of logged motion versus freezing the first future box.

Uses privileged annotation geometry, not a fair input-matched predictor. No
claim of physical safety, causal identification or a new benchmark contribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cavla.geometric_learning import query_curve
from cavla.nuscenes_geometry import signed_point_box_distance


def audit(data,geometry):
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((data/'objects.json').read_text())
    geom={(r['scene'],r['stem']):r for r in json.loads((geometry/'records.json').read_text())}
    static=np.zeros((len(objects),len(d['queries'])),bool);motion={};rows=[]
    for i in np.unique(d['scene_object_ids'][d['scene_object_ids']>=0]):
        obj=objects[i];r=geom[obj['scene'],obj['stem']];centers=np.array(r['agent_centers_xy'])
        displacement=float(np.linalg.norm(centers-centers[:1],axis=-1).max());motion[int(i)]=displacement
        for j,(v,k,h) in enumerate(d['queries']):
            distance=signed_point_box_distance(query_curve(float(v)*15,float(k)*.03),np.tile(centers[0],(64,1)),
                np.repeat(r['agent_yaws'][0],64),np.tile(r['agent_sizes'][0],(64,1)))
            static[i,j]=distance[:round(float(h)*6.4*10)].min()<=2
        rows.append({'scene':obj['scene'],'stem':obj['stem'],'max_displacement_from_tplus0p1_m':displacement})
    ids=d['scene_object_ids'];safe=ids.clip(min=0);stationary=static[safe]&(ids>=0)[...,None]
    true=d['object_targets'];valid=ids>=0;displacement=np.array(list(motion.values()))
    case_static=np.einsum('cn,cnq->cq',d['case_keep'].astype(int),stationary[d['case_scenes']].astype(int))>0
    compound=d['case_cardinality']>=2;ood=d['query_modes']=='ood'
    y=d['case_targets'][compound][:,ood];prediction=case_static[compound][:,ood]
    fn=float((~prediction[y]).mean());fp=float(prediction[~y].mean())
    object_disagreement=(stationary!=true)[valid]
    moving=np.array([[motion.get(int(i),0)>2 for i in scene] for scene in ids])&valid
    summary={'distinct_agents':int(valid.sum()),'fraction_displacement_at_most_0p5m':float((displacement<=.5).mean()),
        'fraction_displacement_more_than_2m':float((displacement>2).mean()),
        'all_object_query_label_disagreement':float(object_disagreement.mean()),
        'moving_object_query_label_disagreement':float((stationary!=true)[moving].mean()) if moving.any() else None,
        'compound_ood_label_disagreement':float((case_static!=d['case_targets'])[compound][:,ood].mean()),
        'privileged_frozen_box_compound_ood_balanced_error':(fn+fp)/2,
        'privileged_frozen_box_compound_ood_false_negative_rate':fn,
        'privileged_frozen_box_compound_ood_false_positive_rate':fp,
        'scope':'Post-hoc label-structure diagnostic with true first-future box at t+0.1s, not input-matched prediction'}
    return summary,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    summary,rows=audit(args.data,args.geometry)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(rows,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
