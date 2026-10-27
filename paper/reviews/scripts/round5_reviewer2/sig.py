"""Reviewer 2 round 5: per-sample contraction signature with clip-bootstrap intervals, plus extra spread-matched controls."""
import json
from pathlib import Path
import numpy as np
from cavla import paper_numbers as pn
A=Path('/mnt/datasets/CAVLA/artifacts');names=[f'context_{i:03d}' for i in range(100,160)]
REF=pn.REFERENCE;TAUS=pn.TAUS;CURVE=pn.CURVE
MODELS=[('alpamayo15','temperature_probe_v1_20261022'),('alpamayo_r1','temperature_probe_r1_v1_20261022'),('alpamayo2','temperature_probe_a2_v1_20261022')]
def ratio(P,n,s,x):
    e1=P[f'{n}_s{s}_euler1_t1.0'][:,:2];a=P[f'{n}_s{s}_{REF}'][:,:2]
    return np.linalg.norm(x[:,:2]-e1)/max(np.linalg.norm(a-e1),1e-9)
data={}
for key,probe in MODELS:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z:P={k:z[k] for k in z.files}
    pn.add_shrinkage(P,names,seeds)
    # extra controls: heterogeneous radial contraction toward E1, g = kappa*d^(gamma-1) capped at 1; and shrink toward mean of A1 samples
    def radial(n,s,kap,gam=0.5):
        e1=P[f'{n}_s{s}_euler1_t1.0'];a=P[f'{n}_s{s}_{REF}'];d=np.linalg.norm(a[:,:2]-e1[:,:2])
        return e1+min(1.,kap*d**(gam-1))*(a-e1)
    meanA={n:np.mean([P[f'{n}_s{s}_{REF}'] for s in seeds],0) for n in names}
    def toMean(n,s,c):a=P[f'{n}_s{s}_{REF}'];return meanA[n]+c*(a-meanA[n])
    for k in (4,10):
        T=np.mean([pn.spread([P[f'{n}_s{s}_euler{k}_t1.0'] for s in seeds]) for n in names])
        for nm,f,lo,hi in [('radial',radial,0.,20.),('toMean',toMean,0.,1.)]:
            for _ in range(40):
                c=(lo+hi)/2
                if np.mean([pn.spread([f(n,s,c) for s in seeds]) for n in names])<T:lo=c
                else:hi=c
            for n in names:
                for s in seeds:P[f'{n}_s{s}_{nm}{k}_t1.0']=f(n,s,c)
    cfgs=CURVE+['euler4_t1.0','euler10_t1.0','shrink4_t1.0','shrink10_t1.0','radial4_t1.0','radial10_t1.0','toMean4_t1.0','toMean10_t1.0','heun5_t1.0','heun3_t1.0']
    cfgs=[c for c in cfgs if f'{names[0]}_s{seeds[0]}_{c}' in P]
    data[key]={c:{n:(pn.spread([P[f'{n}_s{s}_{c}'] for s in seeds]),np.array([ratio(P,n,s,P[f'{n}_s{s}_{c}']) for s in seeds])) for n in names} for c in cfgs}
def stats(tab,sample,c):
    sp=np.array([np.mean([tab[g][n][0] for n in sample]) for g in CURVE]);o=np.argsort(sp)
    s=np.mean([tab[c][n][0] for n in sample]);r=np.concatenate([tab[c][n][1] for n in sample])
    cur=[np.concatenate([tab[g][n][1] for n in sample]) for g in CURVE]
    tm=np.interp(s,sp[o],np.array([np.median(x) for x in cur])[o]);tf=np.interp(s,sp[o],np.array([np.mean(x<1) for x in cur])[o])
    return np.median(r),np.mean(r<1),tm,tf
rng=np.random.default_rng(20261026);B=1000
samples=[names]+[list(rng.choice(names,len(names))) for _ in range(B)]
out={}
for key in data:
    tab=data[key]
    for k in (4,10):
        rows={c:np.array([stats(tab,smp,c) for smp in samples]) for c in [f'euler{k}_t1.0',f'shrink{k}_t1.0',f'radial{k}_t1.0',f'toMean{k}_t1.0']}
        eu=rows[f'euler{k}_t1.0']
        q=lambda v:(v[0],*np.quantile(v[1:],[.025,.975]))
        print(f'== {key} E{k}')
        print('  Euler median %.3f [%.3f,%.3f]  frac %.3f [%.3f,%.3f]'%(*q(eu[:,0]),*q(eu[:,1])))
        print('  Temp@spread median %.3f [%.3f,%.3f]  frac %.3f [%.3f,%.3f]'%(*q(eu[:,2]),*q(eu[:,3])))
        print('  Euler-Temp median %+.3f [%+.3f,%+.3f]  frac %+.3f [%+.3f,%+.3f]'%(*q(eu[:,0]-eu[:,2]),*q(eu[:,1]-eu[:,3])))
        for c in list(rows)[1:]:
            r=rows[c]
            print(f'  {c:14s} median %.3f [%.3f,%.3f] frac %.3f | Euler-ctrl median %+.3f [%+.3f,%+.3f] | ctrl-Temp %+.3f [%+.3f,%+.3f]'%(*q(r[:,0]),r[0,1],*q(eu[:,0]-r[:,0]),*q(r[:,0]-r[:,2])))
        # quantile profile of ratios, point estimate
        for c in [f'euler{k}_t1.0',f'radial{k}_t1.0',f'toMean{k}_t1.0']:
            r=np.concatenate([tab[c][n][1] for n in names]);print('   q10/25/50/75/90',c,np.round(np.quantile(r,[.1,.25,.5,.75,.9]),2))
        # temperature quantile profile at nearest taus
        for g in CURVE[2:5]:
            r=np.concatenate([tab[g][n][1] for n in names]);print('   q10/25/50/75/90',g,np.round(np.quantile(r,[.1,.25,.5,.75,.9]),2))
