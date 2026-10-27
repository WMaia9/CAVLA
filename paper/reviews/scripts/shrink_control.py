"""Reviewer check: is Euler-k distinguishable from linear shrinkage toward E1 (no integrator at all)?
S_c(z) = E1 + c (A1(z) - E1), c matched to the Euler-k spread. Compare flips/ADE1 with E_k and with A_tau_eff,
and per-sample distances E_k(z)->A_tau_eff(z) vs E_k(z)->S_c(z). Joint clip bootstrap over the 3 models."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.trajectory_decision_gate import decisions
A=Path('/mnt/datasets/CAVLA/artifacts'); D=A/'physicalai_test_contexts_v1_20261022'
MODELS=[('A1.5','temperature_probe_v1_20261022'),('R1','temperature_probe_r1_v1_20261022'),('A2','temperature_probe_a2_v1_20261022')]
TAUS=[1.,.9,.8,.7,.6,.5,.4]
rec={r['name']:r for r in json.loads((D/'records.json').read_text()) if 'error' not in r}
names=[f'context_{i:03d}' for i in range(100,160)]
key=lambda x:(x['longitudinal'],x['lateral'])
def spread(ts): return np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(ts) for b in ts[i+1:]])
out={}
for mname,probe in MODELS:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z: P={k:z[k] for k in z.files}
    per=collections.defaultdict(dict)
    # global spreads to set tau_eff and c (paper uses sample-mean spreads)
    sp=lambda cfg:np.mean([spread([P[f'{n}_s{s}_{cfg}'] for s in seeds]) for n in names])
    curve=np.array([sp(f'heun32_t{t}') for t in TAUS])
    for k in [4,10]:
        se=sp(f'euler{k}_t1.0'); o=np.argsort(curve); teff=float(np.interp(se,curve[o],np.array(TAUS)[o]))
        c=se/curve[0]
        lo=max(t for t in TAUS if t<=teff+1e-9); hi=min(t for t in TAUS if t>=teff-1e-9)
        w=0 if hi==lo else (teff-lo)/(hi-lo)
        rows=[]
        for n in names:
            sp_=rec[n]['speed_mps']; hum=np.load(D/f'{n}.npz')['ego_future_xyz']
            E1=np.mean([P[f'{n}_s{s}_euler1_t1.0'] for s in seeds],0)
            r=dict(fE=0,fS=0,fM=0,aE=0,aS=0,aM=0,dEM=0,dES=0,dEA1=0)
            for s in seeds:
                A1=P[f'{n}_s{s}_heun32_t1.0']; Ek=P[f'{n}_s{s}_euler{k}_t1.0']
                At=(1-w)*P[f'{n}_s{s}_heun32_t{lo}']+w*P[f'{n}_s{s}_heun32_t{hi}']  # interpolated plan for distance only
                Sc=E1+c*(A1-E1)
                ref=key(decisions(A1,sp_))
                nearest=P[f'{n}_s{s}_heun32_t{lo if w<.5 else hi}']
                r['fE']+=key(decisions(Ek,sp_))!=ref; r['fS']+=key(decisions(Sc,sp_))!=ref; r['fM']+=key(decisions(nearest,sp_))!=ref
                ade=lambda t:np.linalg.norm(t[:,:2]-hum[:,:2],axis=1).mean()
                r['aE']+=ade(Ek); r['aS']+=ade(Sc); r['aM']+=ade(At)
                d=lambda a,b:np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean()
                r['dEM']+=d(Ek,At); r['dES']+=d(Ek,Sc); r['dEA1']+=d(Ek,A1)
            rows.append({q:v/len(seeds) for q,v in r.items()})
        out[(mname,k)]=dict(teff=teff,c=c,rows=rows)
rng=np.random.default_rng(1); B=2000; idx=[np.arange(60)]+[rng.integers(0,60,60) for _ in range(B)]
for (m,k),v in out.items():
    R={q:np.array([r[q] for r in v['rows']]) for q in v['rows'][0]}
    def stat(f):
        vals=[f({q:a[i] for q,a in R.items()}) for i in idx]; return vals[0],*np.quantile(vals[1:],[.025,.975])
    print(f'{m} E{k} tau_eff {v["teff"]:.2f} c {v["c"]:.2f}')
    print('  flips E/nearestA_tau/shrink: %.3f %.3f %.3f'%(R['fE'].mean(),R['fM'].mean(),R['fS'].mean()))
    print('  flip E-shrink %.3f [%.3f,%.3f]'%stat(lambda d:(d['fE']-d['fS']).mean()))
    print('  ADE1 E/A_tau(interp plan)/shrink: %.3f %.3f %.3f'%(R['aE'].mean(),R['aM'].mean(),R['aS'].mean()))
    print('  ADE E-shrink %.3f [%.3f,%.3f]'%stat(lambda d:(d['aE']-d['aS']).mean()))
    print('  per-sample dist E->A_tau %.3f  E->shrink %.3f  E->A1 %.3f ; diff(A_tau - shrink) %.3f [%.3f,%.3f]'%(R['dEM'].mean(),R['dES'].mean(),R['dEA1'].mean(),*stat(lambda d:(d['dEM']-d['dES']).mean())))
