"""Reviewer-2 round-3 checks: displacement-matched control, shrink-to-mean control, fraction-explained CIs."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/datasets/CAVLA/src')
from cavla.trajectory_decision_gate import decisions
A=Path('/mnt/datasets/CAVLA/artifacts');D=A/'physicalai_test_contexts_v1_20261022'
MODELS=[('alpamayo15','temperature_probe_v1_20261022'),('alpamayo_r1','temperature_probe_r1_v1_20261022'),('alpamayo2','temperature_probe_a2_v1_20261022')]
TAUS=[1.,.9,.8,.7,.6,.5,.4];names=[f'context_{i:03d}' for i in range(100,160)]
rec={r['name']:r for r in json.loads((D/'records.json').read_text()) if 'error' not in r}
key=lambda x:(x['longitudinal'],x['lateral'])
out={}
for mk,probe in MODELS:
    seeds=json.loads((A/probe/'manifest.json').read_text())['seeds']
    with np.load(A/probe/'plans.npz') as z:P={k:z[k] for k in z.files}
    cfgs=sorted({k.split('_',3)[-1] for k in P if '_s92' in k})
    # per clip, per config: displacement from A_ref (mean xy ADE between S(z) and A1(z)), flips, spread
    per={}
    for n in names:
        sp=rec[n]['speed_mps'];ref=[P[f'{n}_s{s}_heun32_t1.0'] for s in seeds];rd=[key(decisions(r,sp)) for r in ref]
        e1=[P[f'{n}_s{s}_euler1_t1.0'] for s in seeds]
        def stats(trajs):
            d=[key(decisions(t,sp)) for t in trajs]
            disp=np.mean([np.linalg.norm(t[:,:2]-r[:,:2],axis=1).mean() for t,r in zip(trajs,ref)])
            spread=np.mean([np.linalg.norm(a[:,:2]-b[:,:2],axis=1).mean() for i,a in enumerate(trajs) for b in trajs[i+1:]])
            return dict(disp=disp,flip=np.mean([a!=b for a,b in zip(d,rd)]),spread=spread,dec=d)
        row={c:stats([P[f'{n}_s{s}_{c}'] for s in seeds]) for c in cfgs}
        for c in np.round(np.arange(0.3,1.0001,0.05),2):  # shrink-to-E1 control in trajectory space
            row[f'shrink{c:.2f}']=stats([m+c*(r-m) for m,r in zip(e1,ref)])
        per[n]=row
    allc=list(per[names[0]].keys())
    mean=lambda c,m,sample=names:float(np.mean([per[n][c][m] for n in sample]))
    curve=[f'heun32_t{t}' for t in TAUS]
    res={}
    def interp(xc,xm,yc,ym,sample=names):  # value of curve metric ym at sampler's xm, along heun32 curve
        xs=np.array([mean(c,xm,sample) for c in curve]);ys=np.array([mean(c,ym,sample) for c in curve]);o=np.argsort(xs)
        return float(np.interp(mean(xc,xm,sample),xs[o],ys[o]))
    for c in [x for x in allc if x.startswith(('euler4','euler10','heun3_t1','heun5_t1','euler6_t1','shrink'))]:
        res[c]=dict(spread=mean(c,'spread'),disp=mean(c,'disp'),flip=mean(c,'flip'),
            flip_at_spread=interp(c,'spread',c,'flip'),flip_at_disp=interp(c,'disp',c,'flip'),
            disp_at_spread=interp(c,'spread',c,'disp'))
    res['curve']={c:dict(spread=mean(c,'spread'),disp=mean(c,'disp'),flip=mean(c,'flip')) for c in curve}
    # bootstrap (joint seeds via rng reused across models) for heun disp-matched residual and fraction explained
    rng=np.random.default_rng(1);B=[list(rng.choice(names,60)) for _ in range(1000)]
    boot={}
    for c in ['euler4_t1.0','euler10_t1.0','heun3_t1.0','heun5_t1.0']:
        if c not in allc: continue
        rd_=[];fe_ade=[];fe_flip=[]
        for s in B:
            rd_.append(mean(c,'flip',s)-interp(c,'disp',c,'flip',s))
            fe_flip.append(1-(mean(c,'flip',s)-interp(c,'spread',c,'flip',s))/max(mean(c,'flip',s),1e-9))
        boot[c]=dict(flip_resid_disp_matched=[res[c]['flip']-res[c]['flip_at_disp'],*np.quantile(rd_,[.025,.975]).tolist()],
            frac_explained_flip_CI=np.quantile(fe_flip,[.025,.975]).tolist())
    res['boot']=boot
    out[mk]=res
    print('==',mk)
    for c,v in res.items():
        if c in('curve','boot'):continue
        print(f"  {c:14s} spread {v['spread']:.3f} disp {v['disp']:.3f} (curve disp at same spread {v['disp_at_spread']:.3f}) flip {100*v['flip']:.1f} | curve flip @spread {100*v['flip_at_spread']:.1f} @disp {100*v['flip_at_disp']:.1f}")
    for c,v in res['curve'].items():print(f"  curve {c:12s} spread {v['spread']:.3f} disp {v['disp']:.3f} flip {100*v['flip']:.1f}")
    print('  boot',json.dumps(boot))
json.dump(out,open(Path(__file__).parent/'heun_control.json','w'),indent=1,default=float)
