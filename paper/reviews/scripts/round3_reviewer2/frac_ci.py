import sys,argparse,numpy as np
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from pathlib import Path
import cavla.paper_numbers as pn
a=argparse.Namespace(data=Path('/mnt/datasets/CAVLA/artifacts/physicalai_test_contexts_v1_20261022'),artifacts=Path('/mnt/datasets/CAVLA/artifacts'),range=[100,160])
names,tables=pn.load(a);rng=np.random.default_rng(20261026);B=[list(rng.choice(names,60)) for _ in range(1000)]
pool={}
for key,_,_ in pn.MODELS:
  for s in ['euler4_t1.0','euler10_t1.0']:
    runs=[pn.matched(tables[key],b,s) for b in B];pt=pn.matched(tables[key],names,s)
    for m in ['ade1','flip']:
      fe=[1-r[m]['residual']/r[m]['effect'] for r in runs]
      print(key,s,m,'expl %.2f CI [%.2f, %.2f]'%(1-pt[m]['residual']/pt[m]['effect'],*np.quantile(fe,[.025,.975])))
    pool.setdefault(s,[]).append([r['flip']['residual'] for r in runs]+[pt['flip']['residual']])
for s,v in pool.items():
  v=np.array(v);m=v.mean(0);print('pooled flip residual',s,'%.2f pp [%.2f, %.2f]'%(100*m[-1],*(100*np.quantile(m[:-1],[.025,.975]))))
