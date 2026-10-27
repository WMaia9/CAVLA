import pickle,numpy as np,sys,collections,json
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
import cavla.matched_temperature_stats as MS; MS.METRICS.append("minority"); summarize=MS.summarize
from cavla.trajectory_decision_gate import decisions
S='/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/'
T,LON,names=pickle.load(open(S+'t.pkl','rb'))
curve=[f'heun32_t{t}' for t in ['1.0','0.9','0.8','0.7','0.6','0.5','0.4']];taus=[1,.9,.8,.7,.6,.5,.4]
rng=np.random.default_rng(1)
# Heun-k negative control (A1.5 only)
for samp in []:
    p=summarize(T['a15'],names,curve,taus,samp);b=[summarize(T['a15'],list(rng.choice(names,60)),curve,taus,samp) for _ in range(500)]
    print(samp,'tau_eff %.3f'%p['tau_eff'],{m:(round(p[m]['effect'],3),round(p[m]['residual'],3),np.round(np.quantile([x[m]['residual'] for x in b],[.025,.975]),3).tolist()) for m in ['ade1','flip','entropy','minade6']})
# minority-decision rate (joint lon x lat key) using full plans
R='/mnt/datasets/CAVLA/artifacts/';data=R+'physicalai_test_contexts_v1_20261022'
rec={r['name']:r for r in json.load(open(data+'/records.json')) if 'error' not in r}
probes={'a15':'temperature_probe_v1_20261022','r1':'temperature_probe_r1_v1_20261022','a2':'temperature_probe_a2_v1_20261022'}
seeds=[9201+i for i in range(6)]
MIN={}
for m,p in probes.items():
    z=np.load(R+p+'/plans.npz')
    cfgs=curve+['euler10_t1.0','euler4_t1.0','euler6_t1.0']
    D={c:{n:[(lambda d:(d['longitudinal'],d['lateral']))(decisions(z[f'{n}_s{s}_{c}'],rec[n]['speed_mps'])) for s in seeds] for n in names} for c in cfgs}
    # mode from pooled reference curve samples (all tau) to avoid using any Euler samples
    mode={n:collections.Counter(sum([D[c][n] for c in curve],[])).most_common(1)[0][0] for n in names}
    MIN[m]={c:{n:np.mean([x!=mode[n] for x in D[c][n]]) for n in names} for c in cfgs}
    for n in names:
        for c in cfgs: T[m][c][n]['minority']=MIN[m][c][n]
for samp in ['euler4_t1.0','euler6_t1.0','euler10_t1.0']:
    per=[summarize(T[m],names,curve,taus,samp)['minority'] for m in T]
    pt=np.mean([x['residual'] for x in per])
    jb=[np.mean([summarize(T[m],idx,curve,taus,samp)['minority']['residual'] for m in T]) for idx in [list(rng.choice(names,60)) for _ in range(1000)]]
    print(samp,'minority eff',[round(x['effect'],4) for x in per],'res',[round(x['residual'],4) for x in per],'pooled res %.4f joint95'%pt,np.round(np.quantile(jb,[.025,.975]),4),'P>=0 %.3f'%np.mean(np.array(jb)>=0))
# curve of minority rate vs tau
for m in T: print(m,[round(np.mean([MIN[m][c][n] for n in names]),3) for c in curve])
