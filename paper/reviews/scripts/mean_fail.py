"""Clips where the one-step mean (E1) / E4 produce a decision no reference seed makes, or erase a minority mode."""
import json,collections
from pathlib import Path
import numpy as np
from cavla.trajectory_decision_gate import decisions
A=Path('artifacts');D=A/'physicalai_test_contexts_v1_20261022'
rec={r['name']:r for r in json.loads((D/'records.json').read_text()) if 'error' not in r}
names=[f'context_{i:03d}' for i in range(100,160)]
key=lambda x:(x['longitudinal'],x['lateral'])
for m,probe in [('A1.5','temperature_probe_v1_20261022'),('R1','temperature_probe_r1_v1_20261022'),('A2','temperature_probe_a2_v1_20261022')]:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z:P={k:z[k] for k in z.files}
    novel=[];erased=[];multi=0
    for n in names:
        sp=rec[n]['speed_mps'];hum=np.load(D/f'{n}.npz')['ego_future_xyz']
        ref=[key(decisions(P[f'{n}_s{s}_heun32_t1.0'],sp)) for s in seeds]
        if len(set(ref))<2:continue
        multi+=1
        for k in (1,4):
            ek=[key(decisions(P[f'{n}_s{s}_euler{k}_t1.0'],sp)) for s in seeds]
            nov=set(ek)-set(ref)
            if nov:novel.append((n,k,dict(collections.Counter(ref)),dict(collections.Counter(ek))))
            lost=set(ref)-set(ek)
            if lost and k==4:
                ade=lambda t:np.linalg.norm(t[:,:2]-hum[:,:2],axis=1).mean()
                erased.append((n,dict(collections.Counter(ref)),dict(collections.Counter(ek)),round(sp,1)))
    print(f'== {m}: {multi} clips with >1 reference decision; E_k decision unseen in reference: {len(novel)}; E4 erases a reference mode: {len(erased)}')
    for x in novel[:8]:print('  novel',x)
    for x in erased[:8]:print('  erased',x)
