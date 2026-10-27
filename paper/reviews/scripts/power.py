import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla import paper_numbers as pn
A=Path('/mnt/datasets/CAVLA/artifacts')
args=argparse.Namespace(data=A/'physicalai_test_contexts_v1_20261022',artifacts=A,range=[100,160])
names,tables=pn.load(args)
res={}
for key in tables:
    t=tables[key]
    for k in [4,10]:
        m=pn.matched(t,names,f'euler{k}_t1.0'); te=m['tau_eff']
        for met in ['flip','ade1','stops']:
            curve=np.array([[t[c][n][met] for c in pn.CURVE] for n in names],float)
            if met=='stops': curve/=6
            at=np.array([np.interp(te,pn.TAUS[::-1],row[::-1]) for row in curve])
            s=np.array([t[f'euler{k}_t1.0'][n][met] for n in names],float)
            if met=='stops': s/=6
            res[(key,k,met)]=s-at
# per-model SD, cross-model corr, pooled SD
for k in [4,10]:
  for met,M in [('flip',.02),('ade1',.05)]:
    R=np.array([res[(key,k,met)] for key in tables])
    sd=R.std(1,ddof=1); C=np.corrcoef(R); pooled=R.mean(0); psd=pooled.std(ddof=1)
    print(f'E{k} {met}: mean resid per model {np.round(R.mean(1),4)} SD {np.round(sd,3)} corr offdiag {np.round(C[np.triu_indices(3,1)],2)} pooled mean {pooled.mean():.4f} pooled SD {psd:.3f}')
    for lab,S in [('per-model(max SD)',sd.max()),('pooled',psd)]:
      for n in [60,200,300,436,600]:
        se=S/np.sqrt(n)
        pw=lambda d:max(0,norm.cdf((M-d)/se-1.645)+norm.cdf((M+d)/se-1.645)-1)
        print(f'    {lab:18s} n={n:4d} SE={se:.4f} power@d=0 {pw(0):.2f} @d=M/4 {pw(M/4):.2f} @d=M/2 {pw(M/2):.2f}')
    for d in [0,.25,.5]:
      for lab,S in [('max-model',sd.max()),('pooled',psd)]:
        z=1.645+norm.ppf(.9 if d>0 else .95); n=(S*z/(M*(1-d)))**2
        print(f'    n for 90% power, true resid={d}*margin, {lab}: {int(np.ceil(n))}')
st=np.array([res[(key,4,'stops')] for key in tables]); print('stop resid E4 per clip nonzero clips',(np.abs(st).sum(0)>0).sum())
