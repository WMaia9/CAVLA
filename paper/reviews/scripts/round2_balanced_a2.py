import pickle,json,numpy as np,sys
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.matched_temperature_stats import summarize
S='/tmp/claude-1000/-mnt-datasets-CAVLA/0176a190-3508-42e5-a6ef-0552d3713372/scratchpad/'
T,LON,names=pickle.load(open(S+'t.pkl','rb'))
data='/mnt/datasets/CAVLA/artifacts/physicalai_test_contexts_v1_20261022'
rec={r['name']:r for r in json.load(open(data+'/records.json')) if 'error' not in r}
for n in ['context_113','context_141','context_127','context_124']: print(n,{k:v for k,v in rec[n].items() if not isinstance(v,(list,dict))})
for m in LON:
  for c in ['heun32_t1.0','heun32_t0.6','euler10_t1.0','euler4_t1.0','euler1_t1.0']:
    print(m,c,'113',LON[m][c]['context_113'],'141',LON[m][c]['context_141'])
curve=[f'heun32_t{t}' for t in ['1.0','0.9','0.8','0.7','0.6','0.5','0.4']];taus=[1,.9,.8,.7,.6,.5,.4]
rng=np.random.default_rng(0)
for samp in ['euler4_t1.0','euler10_t1.0']:
  for metric in ['stop','entropy','minade6','flip']:
    pt=np.mean([summarize(T[m],names,curve,taus,samp)[metric]['residual'] for m in T])
    joint=[];ind=[]
    for b in range(1000):
        idx=list(rng.choice(names,60));joint.append(np.mean([summarize(T[m],idx,curve,taus,samp)[metric]['residual'] for m in T]))
        ind.append(np.mean([summarize(T[m],list(rng.choice(names,60)),curve,taus,samp)[metric]['residual'] for m in T]))
    print(samp,metric,'pooled %.4f'%pt,'joint95',np.round(np.quantile(joint,[.025,.975]),4),'indep95',np.round(np.quantile(ind,[.025,.975]),4),'P(joint>=0)=%.3f'%np.mean(np.array(joint)>=0))
# leave-one-clip-out for stop
for drop in ['context_113','context_141']:
    idx=[n for n in names if n!=drop]
    print('drop',drop,[round(np.mean([summarize(T[m],idx,curve,taus,s)['stop']['residual'] for m in T])*100,2) for s in ['euler4_t1.0','euler10_t1.0']])
