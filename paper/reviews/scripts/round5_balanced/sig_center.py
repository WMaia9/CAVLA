"""Is the per-sample signature driven by E1's residual seed spread? Recompute contraction ratios toward the
seed-mean of E1 (a common center), with shrink toward that common center (spread-matched by construction)."""
import json,numpy as np
from pathlib import Path
A=Path('artifacts');names=[f'context_{i:03d}' for i in range(100,160)];TAUS=[1.,.9,.8,.7,.6,.5,.4]
MODELS=[('A1.5','temperature_probe_v1_20261022'),('R1','temperature_probe_r1_v1_20261022'),('A2','temperature_probe_a2_v1_20261022')]
nrm=lambda a,b:np.linalg.norm(a[:,:2]-b[:,:2])
def spread(ts):return np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(ts) for b in ts[i+1:]])
for m,probe in MODELS:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z:P={k:z[k] for k in z.files}
    C={n:np.mean([P[f'{n}_s{s}_euler1_t1.0'] for s in seeds],0) for n in names}
    def ratios(get,center):return np.array([nrm(get(n,s),center(n,s))/max(nrm(P[f'{n}_s{s}_heun32_t1.0'],center(n,s)),1e-9) for n in names for s in seeds])
    cm=lambda n,s:C[n]; cs=lambda n,s:P[f'{n}_s{s}_euler1_t1.0']
    sp=lambda cfg:np.mean([spread([P[f'{n}_s{s}_{cfg}'] for s in seeds]) for n in names])
    curve_sp=np.array([sp(f'heun32_t{t}') for t in TAUS]);o=np.argsort(curve_sp)
    for center,lab in [(cs,'per-seed E1'),(cm,'mean E1')]:
        med=np.array([np.median(ratios(lambda n,s,t=t:P[f'{n}_s{s}_heun32_t{t}'],center)) for t in TAUS])
        fr=np.array([np.mean(ratios(lambda n,s,t=t:P[f'{n}_s{s}_heun32_t{t}'],center)<1) for t in TAUS])
        out=[]
        for k in [4,10]:
            s=sp(f'euler{k}_t1.0');rE=ratios(lambda n,s_:P[f'{n}_s{s_}_euler{k}_t1.0'],center)
            c=s/curve_sp[TAUS.index(1.)]  # common-center shrink: spread scales exactly by c
            out.append(f'E{k}: Euler {np.median(rE):.3f} ({100*np.mean(rE<1):.0f}%)  Temp {np.interp(s,curve_sp[o],med[o]):.3f} ({100*np.interp(s,curve_sp[o],fr[o]):.0f}%)  shrink-to-mean c={c:.3f}')
        print(m,lab,' | '.join(out))
