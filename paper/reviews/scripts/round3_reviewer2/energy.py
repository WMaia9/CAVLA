import json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.trajectory_decision_gate import decisions
A=Path('/mnt/datasets/CAVLA/artifacts');D=A/'physicalai_test_contexts_v1_20261022'
rec={r['name']:r for r in json.loads((D/'records.json').read_text()) if 'error' not in r}
names=[f'context_{i:03d}' for i in range(100,160)];key=lambda x:(x['longitudinal'],x['lateral'])
d=lambda a,b:np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean()
def ed(X,Y):
    return 2*np.mean([d(x,y) for x in X for y in Y])-np.mean([d(a,b) for a in X for b in X])-np.mean([d(a,b) for a in Y for b in Y])
def tv(X,Y,sp):
    from collections import Counter
    cx=Counter(key(decisions(x,sp)) for x in X);cy=Counter(key(decisions(y,sp)) for y in Y)
    return .5*sum(abs(cx[k]-cy[k]) for k in set(cx)|set(cy))/len(X)
for mk,probe in [('alpamayo15','temperature_probe_v1_20261022'),('alpamayo_r1','temperature_probe_r1_v1_20261022'),('alpamayo2','temperature_probe_a2_v1_20261022')]:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z:P={k:z[k] for k in z.files}
    S=lambda n,c:[P[f'{n}_s{s}_{c}'] for s in seeds]
    pairs=[('euler10_t1.0','heun32_t0.8'),('euler10_t1.0','heun32_t0.9'),('heun32_t0.8','heun32_t0.9'),
           ('euler4_t1.0','heun32_t0.6'),('euler4_t1.0','heun32_t0.7'),('heun32_t0.6','heun32_t0.7'),('heun32_t0.5','heun32_t0.7')]
    if mk=='alpamayo15':pairs+=[('heun5_t1.0','heun32_t1.0'),('heun5_t1.0','heun32_t0.9'),('heun32_t0.9','heun32_t1.0')]
    print('==',mk)
    for a,b in pairs:
        e=[ed(S(n,a),S(n,b)) for n in names];t=[tv(S(n,a),S(n,b),rec[n]['speed_mps']) for n in names]
        print(f'  {a:13s} vs {b:12s} energy {np.mean(e):.3f} (median {np.median(e):.3f})  decision-TV {100*np.mean(t):.1f}%')
