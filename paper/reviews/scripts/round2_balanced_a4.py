import numpy as np,sys,json
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.trajectory_decision_gate import decisions
R='/mnt/datasets/CAVLA/artifacts/';data=R+'physicalai_test_contexts_v1_20261022'
rec={r['name']:r for r in json.load(open(data+'/records.json')) if 'error' not in r}
probes={'a15':'temperature_probe_v1_20261022','r1':'temperature_probe_r1_v1_20261022','a2':'temperature_probe_a2_v1_20261022'}
names=[f'context_{i}' for i in range(100,160)];seeds=[9201+i for i in range(6)]
key=lambda d:(d['longitudinal'],d['lateral'])
rows=[]
for m,p in probes.items():
    z=np.load(R+p+'/plans.npz')
    for n in names:
        for s in seeds:
            g=lambda c:key(decisions(z[f'{n}_s{s}_{c}'],rec[n]['speed_mps']))
            ref,e1=g('heun32_t1.0'),g('euler1_t1.0')
            for samp,match in [('euler4_t1.0','heun32_t0.6'),('euler10_t1.0','heun32_t0.8')]:
                a,b=g(samp),g(match)
                rows.append((m,n,samp,a!=ref,a==e1,b!=ref,b==e1,ref==e1))
import collections
for samp in ['euler4_t1.0','euler10_t1.0']:
    r=[x for x in rows if x[2]==samp]
    ef=[x for x in r if x[3] and not x[7]]; tf=[x for x in r if x[5] and not x[7]]
    print(samp,'Euler flips(ref!=E1 class):',len(ef),'-> to E1 class %.2f'%np.mean([x[4] for x in ef]),'| temp flips:',len(tf),'-> to E1 class %.2f'%np.mean([x[6] for x in tf]),
          '| clips w/ euler flips',len({(x[0],x[1]) for x in ef}),'distinct clips',len({x[1] for x in ef}))
    # among samples where ref != E1 class: P(move to E1 class) under Euler vs temperature; clip-level paired
    d=collections.defaultdict(lambda:[0,0,0])
    for x in r:
        if not x[7]: d[x[1]][0]+=x[4];d[x[1]][1]+=x[6];d[x[1]][2]+=1
    diff=np.array([(v[0]-v[1])/v[2] for v in d.values()])
    rng=np.random.default_rng(0);bs=[rng.choice(diff,len(diff)).mean() for _ in range(4000)]
    print('   per-clip (pooled models) P(to E1 class) Euler - temp: %.3f'%diff.mean(),np.round(np.quantile(bs,[.025,.975]),3),'n clips',len(diff),'clips Euler>temp',int((diff>0).sum()),'<',int((diff<0).sum()))
