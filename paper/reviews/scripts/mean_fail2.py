import json
from pathlib import Path
import numpy as np
from cavla.trajectory_decision_gate import decisions
A=Path('artifacts');D=A/'physicalai_test_contexts_v1_20261022'
rec={r['name']:r for r in json.loads((D/'records.json').read_text()) if 'error' not in r}
names=[f'context_{i:03d}' for i in range(100,160)];key=lambda x:(x['longitudinal'],x['lateral'])
for m,probe in [('A1.5','temperature_probe_v1_20261022'),('R1','temperature_probe_r1_v1_20261022'),('A2','temperature_probe_a2_v1_20261022')]:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z:P={k:z[k] for k in z.files}
    for cfg in ['euler4_t1.0','heun32_t0.6','euler1_t1.0','heun32_t0.4']:
        nov=era=0;novclips=set()
        for n in names:
            sp=rec[n]['speed_mps'];ref=[key(decisions(P[f'{n}_s{s}_heun32_t1.0'],sp)) for s in seeds]
            if len(set(ref))<2:continue
            e=[key(decisions(P[f'{n}_s{s}_{cfg}'],sp)) for s in seeds]
            if set(e)-set(ref):nov+=1;novclips.add(n)
            if set(ref)-set(e):era+=1
        print(m,f'{cfg:12s} clips with a decision absent from reference: {nov:2d} | erasing a reference decision: {era:2d}')
