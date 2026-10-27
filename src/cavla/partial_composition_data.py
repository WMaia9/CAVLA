"""Build partial OR supervision and held-out deletion combinations.

These are algebraic exclusions of selected logged constraints, not edited-image
or reactive physical counterfactual ground truth.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from cavla.constraint_composition import assignments_from_observations


def instance_representatives(scene,members,objects,geometry):
    representatives={}
    for i in members:
        r=geometry[scene,objects[i]['stem']];instance=r['instance_token']
        previous=representatives.get(instance)
        if previous is None or (r['best_iou'],objects[i]['stem'])>(geometry[scene,objects[previous]['stem']]['best_iou'],objects[previous]['stem']):
            representatives[instance]=i
    return representatives


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deduplicate-geometry',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    objects=json.loads((args.data/'objects.json').read_text())
    geometry=None
    if args.deduplicate_geometry:
        geometry={(r['scene'],r['stem']):r for r in json.loads((args.deduplicate_geometry/'records.json').read_text())}
    pairs={r['scene']:r for r in json.loads(args.pairs.read_text())}
    with np.load(args.data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    scenes=sorted({r['scene'] for r in objects});max_n=max(sum(o['scene']==s for o in objects) for s in scenes)
    first=d['object_ids']==0;queries=d['queries'][first];modes=d['query_modes'][first];nq=len(queries)
    ids=np.full((len(scenes),max_n),-1,int);labels=np.zeros((len(scenes),max_n,nq),bool)
    removed=[];scene_rows=[];duplicates=[]
    for si,scene in enumerate(scenes):
        members=[i for i,r in enumerate(objects) if r['scene']==scene]
        representatives={}
        if geometry is not None:
            representatives=instance_representatives(scene,members,objects,geometry)
            kept=set(representatives.values())
            duplicates.extend({'scene':scene,'excluded_object_id':i,'kept_object_id':representatives[geometry[scene,objects[i]['stem']]['instance_token']],
                'instance_token':geometry[scene,objects[i]['stem']]['instance_token']} for i in members if i not in kept)
            members=[i for i in members if i in kept]
        ids[si,:len(members)]=members
        for j,i in enumerate(members):
            use=d['object_ids']==i
            assert np.array_equal(queries,d['queries'][use]) and np.array_equal(modes,d['query_modes'][use])
            labels[si,j]=d['targets'][use]
        target=pairs[scene]['removed_object_id'] if scene in pairs else min(members,
            key=lambda i:hashlib.sha256((scene+'/'+objects[i]['stem']).encode()).hexdigest())
        if target not in members:
            target=representatives[geometry[scene,objects[target]['stem']]['instance_token']]
        removed.append(members.index(target));row=objects[members[0]]
        assert len({objects[i]['fold'] for i in members})==1
        scene_rows.append({'scene':scene,'log_token':row['log_token'],'fold':row['fold'],
            'n_objects':len(members),'removed_object_id':target,'removed_local_index':members.index(target)})
    removed=np.array(removed);weak=[];case_scenes=[];keep=[];cardinalities=[];targets=[];lower=[];upper=[]
    for si,scene in enumerate(scenes):
        n=scene_rows[si]['n_objects'];base=labels[si,:n];count=base.sum(0)
        weak.append(np.stack([count>0,count-base[removed[si]]>0],-1))
        subsets=[()]+[(j,) for j in range(n)]
        for k in [2,3]:
            if k<n:subsets.extend(itertools.combinations(range(n),k))
        masks=np.zeros((len(subsets),max_n),bool);masks[:,:n]=True
        for j,subset in enumerate(subsets):masks[j,list(subset)]=False
        truth=(masks[:,:n].astype(int)@base.astype(int))>0
        lo=np.zeros_like(truth);hi=np.zeros_like(truth)
        pair=np.stack([count>0,count-base[removed[si]]>0],-1)
        for pattern in np.unique(pair,axis=0):
            observed=[None]*n;observed[removed[si]]=bool(pattern[1])
            possible=assignments_from_observations(bool(pattern[0]),observed)
            outcomes=(masks[:,:n].astype(int)@possible.astype(int).T)>0
            cols=np.all(pair==pattern,axis=1)
            lo[:,cols]=outcomes.min(1)[:,None];hi[:,cols]=outcomes.max(1)[:,None]
        assert np.all(lo<=truth) and np.all(truth<=hi)
        case_scenes.extend([si]*len(subsets));keep.extend(masks);cardinalities.extend(map(len,subsets))
        targets.extend(truth);lower.extend(lo);upper.extend(hi)
    np.savez_compressed(args.out/'dataset.npz',scene_object_ids=ids,object_targets=labels,
        queries=queries,query_modes=modes,removed_index=removed,weak_labels=np.array(weak),
        case_scenes=np.array(case_scenes),case_keep=np.array(keep),case_cardinality=np.array(cardinalities),
        case_targets=np.array(targets),oracle_lower=np.array(lower),oracle_upper=np.array(upper),
        scene_folds=np.array([r['fold'] for r in scene_rows]))
    (args.out/'scenes.json').write_text(json.dumps(scene_rows,indent=2))
    (args.out/'objects.json').write_text(json.dumps(objects,indent=2))
    manifest={'scope':'Simulated weak supervision from annotated logged constraints; not actual compound edit images',
        'parent':str(args.data),'pairs':str(args.pairs),'scenes':len(scenes),'objects':len(objects),
        'observed_train_bits_before_outer_split':int(len(scenes)*(modes=='id').sum()*2),
        'evaluation_cases_by_cardinality':{str(k):int(np.sum(np.array(cardinalities)==k)) for k in range(4)},
        'scene_queries':len(scenes)*nq,'evaluation_query_cases':len(case_scenes)*nq,
        'oracle_ambiguity_fraction_compound_ood':float((np.array(lower)!=np.array(upper))[np.array(cardinalities)>=2][:,modes=='ood'].mean()),
        'data_sha256':hashlib.sha256((args.data/'dataset.npz').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_COMPOSICAO_PARCIAL_V1.md').read_bytes()).hexdigest(),
        'selection':'Previously selected removal when available, otherwise first SHA256(scene/stem)',
        'deduplication':{'geometry':str(args.deduplicate_geometry) if args.deduplicate_geometry else None,
            'rule':'At most one detection per physical instance and scene, highest 3D projection IoU then stem tie-break',
            'excluded_detections':duplicates,'distinct_agent_rows':int((ids>=0).sum()),
            'feature_normalization':'Parent original feature rows retained for train-only PCA/normalization in all methods; discarded duplicates are not supervised/pooled as agents'},
        'oracle':'Privileged diagnostic using original+one singleton gold test answer; not network input'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'PROTOCOL.md').write_bytes(Path('docs/PROTOCOLO_COMPOSICAO_PARCIAL_V1.md').read_bytes())
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
