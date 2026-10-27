"""Audit redundant query constraints within the annotated object subset.

OR composition is a fixed logged-geometry abstraction, not reactive scene risk.
The singleton-deletion ambiguity is elementary group testing, not a new theorem.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def singleton_signature(bits):
    bits=np.asarray(bits,dtype=bool)
    total=int(bits.sum())
    return (total>0,*((total-bits.astype(int))>0).tolist())


def assignments_from_observations(original,single_deletions):
    """Enumerate exact OR assignments; None denotes an unobserved outcome."""
    n=len(single_deletions)
    if not 1<=n<=16:raise ValueError('Exact enumeration supports 1..16 agents')
    if original not in (None,False,True,0,1):raise ValueError('Original outcome must be binary or None')
    if any(v not in (None,False,True,0,1) for v in single_deletions):
        raise ValueError('Deletion outcomes must be binary or None')
    assignments=((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(bool)
    total=assignments.sum(1);keep=np.ones(len(assignments),bool)
    if original is not None:keep&=(total>0)==bool(original)
    for j,value in enumerate(single_deletions):
        if value is not None:keep&=((total-assignments[:,j])>0)==bool(value)
    result=assignments[keep]
    if not len(result):raise ValueError('Observations inconsistent with exact OR/frozen-agent assumptions')
    return result


def compatible_assignments(bits):
    signature=singleton_signature(bits)
    return assignments_from_observations(signature[0],signature[1:])


def composition_bounds(original,single_deletions,removed):
    """Identification interval for an unobserved deletion set, not confidence."""
    assignments=assignments_from_observations(original,single_deletions)
    n=assignments.shape[1]
    if any(not isinstance(i,(int,np.integer)) or i<0 or i>=n for i in removed):
        raise ValueError('Removed agent index outside observed set')
    remaining=[i for i in range(n) if i not in removed]
    outcomes=assignments[:,remaining].any(1)
    return {'lower':int(outcomes.min()),'upper':int(outcomes.max()),'compatible_assignments':len(assignments)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    objects=json.loads((args.data/'objects.json').read_text())
    with np.load(args.data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    rows=[];cache={};counts={'scene_queries':0,'positive_scene_queries':0,'redundant_scene_queries':0,
        'positive_object_queries':0,'masked_positive_object_queries':0,'nonidentified_object_queries':0,
        'object_queries':0}
    for scene in sorted({r['scene'] for r in objects}):
        ids=[i for i,r in enumerate(objects) if r['scene']==scene];labels=[];query=None
        for i in ids:
            use=d['object_ids']==i
            if query is None:query=d['queries'][use]
            else:assert np.array_equal(query,d['queries'][use])
            labels.append(d['targets'][use].astype(bool))
        labels=np.stack(labels);multiplicity=labels.sum(0);ambiguities=[]
        for q in range(labels.shape[1]):
            key=(len(ids),tuple(labels[:,q]))
            if key not in cache:
                compatible=compatible_assignments(labels[:,q])
                cache[key]=(len(compatible),int((compatible.min(0)!=compatible.max(0)).sum()))
            ambiguities.append(cache[key])
        ambiguity=np.array(ambiguities)
        row={'scene':scene,'n_objects':len(ids),'n_queries':len(query),
            'positive_scene_queries':int((multiplicity>0).sum()),
            'redundant_scene_queries':int((multiplicity>=2).sum()),
            'positive_object_queries':int(multiplicity.sum()),
            'masked_positive_object_queries':int(multiplicity[multiplicity>=2].sum()),
            'nonidentified_object_queries':int(ambiguity[:,1].sum()),
            'max_blockers':int(multiplicity.max()),
            'max_compatible_assignments_after_all_single_deletions':int(ambiguity[:,0].max())}
        rows.append(row)
        counts['scene_queries']+=len(query);counts['positive_scene_queries']+=row['positive_scene_queries']
        counts['redundant_scene_queries']+=row['redundant_scene_queries']
        counts['positive_object_queries']+=row['positive_object_queries']
        counts['masked_positive_object_queries']+=row['masked_positive_object_queries']
        counts['nonidentified_object_queries']+=row['nonidentified_object_queries']
        counts['object_queries']+=labels.size
    summary={**counts,
        'fraction_positive_scene_queries_with_redundancy':counts['redundant_scene_queries']/counts['positive_scene_queries'],
        'fraction_positive_object_queries_masked_in_single_deletion':counts['masked_positive_object_queries']/counts['positive_object_queries'],
        'scope':'Exact oracle arithmetic in the selected annotated-agent subset; not observed VLA effects or full-scene safety',
        'identification':'Exhaustive assignments under OR composition, observing original plus every single-object deletion; ignores cross-query geometric priors',
        'limitations':['Only the 584 selected agents, omits other scene constraints','Frozen logged agent futures',
            'Elementary group-testing phenomenon, not claimed as novel','No real multiobject removal images tested']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    (args.out/'records.json').write_text(json.dumps(rows,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'data':str(args.data),
        'dataset_sha256':hashlib.sha256((args.data/'dataset.npz').read_bytes()).hexdigest()},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
