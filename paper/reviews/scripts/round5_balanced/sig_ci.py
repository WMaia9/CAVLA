"""Clip-bootstrap CIs for the per-sample signature (Finding 5): median contraction ratio of E_k vs
reference at matched spread (interpolated on tau grid) vs Shrink_k; joint bootstrap over clips, pooled over models."""
import argparse,numpy as np
from pathlib import Path
from cavla import paper_numbers as pn
a=argparse.Namespace(data=Path('artifacts/physicalai_test_contexts_v1_20261022'),artifacts=Path('artifacts'),range=[100,160])
names,tables=pn.load(a)
rng=np.random.default_rng(5);B=1000;idx=[np.arange(60)]+[rng.integers(0,60,60) for _ in range(B)]
TAUS=np.array(pn.TAUS)
def stats(t,ii,k):
    ns=[names[i] for i in ii]
    sp=lambda c:np.mean([t[c][n]['spread'] for n in ns])
    r=lambda c:np.array([x for n in ns for x in t[c][n]['contraction']])
    spreads=np.array([sp(c) for c in pn.CURVE]);o=np.argsort(spreads);s=sp(f'euler{k}_t1.0')
    med=np.array([np.median(r(c)) for c in pn.CURVE]);fr=np.array([np.mean(r(c)<1) for c in pn.CURVE])
    rE=r(f'euler{k}_t1.0');rS=r(f'shrink{k}_t1.0')
    return dict(E=np.median(rE),T=np.interp(s,spreads[o],med[o]),S=np.median(rS),fE=np.mean(rE<1),fT=np.interp(s,spreads[o],fr[o]),fS=np.mean(rS<1),sdS=np.std(rS))
q=lambda x:f'{x[0]:+.3f} [{np.quantile(x[1:],.025):+.3f},{np.quantile(x[1:],.975):+.3f}]'
for k in [4,10]:
    allr=[]
    for key in tables:
        R=[stats(tables[key],ii,k) for ii in idx];allr.append(R)
        g=lambda f:np.array([f(x) for x in R])
        print(key,f'E{k}','E',round(R[0]['E'],3),'T',round(R[0]['T'],3),'S',round(R[0]['S'],3),'sd(S ratios)',round(R[0]['sdS'],4),
              '| E-T',q(g(lambda x:x['E']-x['T'])),'E-S',q(g(lambda x:x['E']-x['S'])),'| frac E-T',q(g(lambda x:x['fE']-x['fT'])))
    P=lambda f:np.mean([[f(x) for x in R] for R in allr],0)
    print(' pooled',f'E{k}','E-T',q(P(lambda x:x['E']-x['T'])),'E-S',q(P(lambda x:x['E']-x['S'])),'|T-S|-|E-T|',q(P(lambda x:abs(x['T']-x['S'])-abs(x['E']-x['T']))),'fracE-T',q(P(lambda x:x['fE']-x['fT'])))
