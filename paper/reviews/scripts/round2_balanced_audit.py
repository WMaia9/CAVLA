import json,collections,numpy as np,sys
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.trajectory_decision_gate import decisions
from cavla.matched_temperature_stats import per_clip,summarize
R='/mnt/datasets/CAVLA/artifacts/'
data=R+'physicalai_test_contexts_v1_20261022'
records={r['name']:r for r in json.load(open(data+'/records.json')) if 'error' not in r}
keep={f'context_{i:03d}' for i in range(100,160)}
probes={'a15':'temperature_probe_v1_20261022','r1':'temperature_probe_r1_v1_20261022','a2':'temperature_probe_a2_v1_20261022'}
seeds=[9201+i for i in range(6)]
T={};curve=[f'heun32_t{t}' for t in ['1.0','0.9','0.8','0.7','0.6','0.5','0.4']];taus=[1,.9,.8,.7,.6,.5,.4]
LON={}
for m,p in probes.items():
    z=np.load(R+p+'/plans.npz');plans={k:z[k] for k in z.files}
    names=sorted(n for n in {k.split('_s')[0] for k in plans if '_s92' in k} if n in keep)
    configs=sorted({k.split('_',3)[-1] for k in plans if '_s92' in k})
    human={n:plans[f'{n}_human'] if f'{n}_human' in plans else np.load(f'{data}/{n}.npz')['ego_future_xyz'] for n in names}
    T[m]=per_clip(plans,names,configs,seeds,'heun32_t1.0',human,{n:records[n]['speed_mps'] for n in names})
    LON[m]={c:{n:[decisions(plans[f'{n}_s{s}_{c}'],records[n]['speed_mps'])['longitudinal'] for s in seeds] for n in names} for c in configs}
    print(m,len(names))
names=sorted(T['a15']['heun32_t1.0'])
import pickle;pickle.dump((T,LON,names),open('/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/t.pkl','wb'))
# 1 which clips stop
for m in T:
    for c in curve+['euler10_t1.0','euler4_t1.0']:
        st={n:sum(x=='stop' for x in LON[m][c][n]) for n in names if any(x=='stop' for x in LON[m][c][n])}
        print(m,c,st)
