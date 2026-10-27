"""Log-held-out onset probes: kinematics, fluctuations, raw past and semantic oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


def randomized_pca(train, all_values, components=16):
    """Train-only randomized range finder; deterministic, never fit on validation/test."""
    mean=train.mean(0);centered=train-mean
    rng=np.random.default_rng(20261021)
    omega=rng.normal(size=(train.shape[1],components+16))
    q,_=np.linalg.qr(centered@omega,mode='reduced')
    for _ in range(2):
        z,_=np.linalg.qr(centered.T@q,mode='reduced')
        q,_=np.linalg.qr(centered@z,mode='reduced')
    _,_,vt=np.linalg.svd(q.T@centered,full_matrices=False)
    return (all_values-mean)@vt[:components].T


def log_mean(values,logs):
    return float(np.mean([values[logs==l].mean() for l in sorted(set(logs))]))


def metrics(y,p,logs):
    p=np.clip(p,1e-8,1-1e-8);n=int(y.sum());m=len(y)-n
    order=np.argsort(-p,kind='stable');precision=np.cumsum(y[order])/np.arange(1,len(y)+1)
    return dict(agents=len(y),events=n,logs=len(set(logs)),log_nll=log_mean(-y*np.log(p)-(1-y)*np.log1p(-p),logs),
        log_brier=log_mean((p-y)**2,logs),auc=float((rankdata(p)[y==1].sum()-n*(n+1)/2)/(n*m)) if n and m else None,
        average_precision=float((precision*y[order]).sum()/max(n,1)),prevalence=float(y.mean()))


def fit_predict(x,y,logs,val,vy,vlogs,ev):
    unique,inverse,counts=np.unique(logs,return_inverse=True,return_counts=True)
    wgt=1/(len(unique)*counts[inverse])
    base=np.clip(np.sum(y*wgt),1e-6,1-1e-6);best=log_mean(-vy*np.log(base)-(1-vy)*np.log1p(-base),vlogs)
    winner=np.full(len(ev),base);selection=dict(model='constant',validation_log_nll=best,candidates=[])
    mean=x.mean(0);scale=x.std(0).clip(min=1e-6)
    transform=lambda a:np.c_[np.ones(len(a)),np.clip((a-mean)/scale,-10,10)]
    xx=transform(x);vv=transform(val);ee=transform(ev)
    for alpha in [.001,.01,.1,1.,10.]:
        def objective(w):
            z=xx@w;p=expit(z);penalty=w.copy();penalty[0]=0
            return np.sum(wgt*(np.logaddexp(0,z)-y*z))+alpha*np.sum(penalty**2)/2,xx.T@(wgt*(p-y))+alpha*penalty
        start=np.zeros(xx.shape[1]);start[0]=np.log(base/(1-base))
        opt=minimize(objective,start,jac=True,method='L-BFGS-B',options=dict(maxiter=250,ftol=1e-10))
        pv=np.clip(expit(vv@opt.x),1e-8,1-1e-8);score=log_mean(-vy*np.log(pv)-(1-vy)*np.log1p(-pv),vlogs)
        selection['candidates'].append(dict(alpha=alpha,validation_log_nll=score,converged=bool(opt.success)))
        if score<best:best=score;winner=expit(ee@opt.x);selection.update(model='logistic',alpha=alpha,validation_log_nll=score)
    return winner,selection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--parked-stopped-only',action='store_true');p.add_argument('--data',type=Path,default=Path('artifacts/motion_transition_inventory_v1_20261021'));p.add_argument('--clip',type=Path)
    p.add_argument('--extra-features',action='append',default=[],help='NAME=DIRECTORY, frozen VLM last-layer ROI/context, PCA16')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=args.data;objects_path=source/('landmarks.json' if (source/'landmarks.json').exists() else 'objects.json');rows=json.loads(objects_path.read_text())
    with np.load(source/'features.npz') as z:arrays={k:z[k] for k in z.files}
    logs=np.array([r['log'] for r in rows]);folds=np.array([int(hashlib.sha256(l.encode()).hexdigest(),16)%5 for l in logs])
    event=np.array([r['event_seconds'] is not None and r['event_seconds']<=4 for r in rows]);follow=np.array([r['followup_seconds'] for r in rows])
    known=event|(follow>=4)
    if args.parked_stopped_only:known &= np.array([r['attribute'] in ['vehicle.parked','vehicle.stopped'] for r in rows])
    strict=np.array([r['strict_stationary'] for r in rows]);rng=np.random.default_rng(20261021)
    category=sorted({r['category'] for r in rows});attribute=sorted({r['attribute'] for r in rows})
    fixed=np.array([[r['category']==k for k in category]+r['size'] for r in rows],float)
    oracle=np.array([[r['attribute']==k for k in attribute] for r in rows],float)
    basic=np.c_[fixed,arrays['basic']]
    if 'relative_geometry' in arrays:basic=np.c_[basic,arrays['relative_geometry']]
    clips={}
    if args.clip:
        with np.load(args.clip/'features.npz') as z:clips={k:z[k].astype(float) for k in ['crop','context','features']}
        manifest=json.loads((args.clip/'manifest.json').read_text());assert manifest['objects_sha256']==hashlib.sha256(objects_path.read_bytes()).hexdigest()
    extras={}
    for spec in args.extra_features:
        name,directory=spec.split('=',1);directory=Path(directory)
        manifest=json.loads((directory/'manifest.json').read_text());assert manifest['objects_sha256']==hashlib.sha256(objects_path.read_bytes()).hexdigest()
        with np.load(directory/'features.npz') as z:
            assert int(z['layers'][-1])==36
            roi=z['features'][:,-1].astype(float);context=z['context'][:,-1].astype(float)
        assert len(roi)==len(rows) and np.isfinite(roi).all() and np.isfinite(context).all()
        for suffix,value in [('roi',roi),('context',context),('both',np.c_[roi,context])]:
            key='raw_'+name+'_'+suffix;assert key not in extras;extras[key]=value
    raw=np.c_[basic,arrays['raw_history']];noise=rng.normal(size=(len(rows),16 if args.clip or extras else 8))
    results={};records=[];save={}
    for cohort,mask in [('principal',known),('strict',known&strict)]:
        ids=np.flatnonzero(mask);ll=logs[ids];ff=folds[ids];y=event[ids].astype(float)
        methods=['constant','basic','raw','raw_fluctuation','raw_noise','raw_rff','raw_oracle']+['raw_clip_'+k for k in clips]+list(extras);pred={m:np.full(len(ids),np.nan) for m in methods}
        for f in range(5):
            tr=np.flatnonzero((ff!=f)&(ff!=(f+1)%5));va=np.flatnonzero(ff==(f+1)%5);te=np.flatnonzero(ff==f)
            assert not(set(ll[tr])&(set(ll[va])|set(ll[te])))
            mu=raw[ids[tr]].mean(0);scale=raw[ids[tr]].std(0).clip(min=1e-6)
            # Fixed random nonlinear map; no full-dataset fitting or hyperparameter search.
            rrng=np.random.default_rng(20261021);w=rrng.normal(size=(raw.shape[1],64))/np.sqrt(raw.shape[1]);phase=rrng.uniform(0,2*np.pi,64)
            rff=np.sqrt(2/64)*np.cos(np.clip((raw[ids]-mu)/scale,-10,10)@w+phase)
            values={'constant':np.zeros((len(ids),1)),'basic':basic[ids],'raw':raw[ids],
                'raw_fluctuation':np.c_[raw[ids],arrays['fluctuation'][ids]],'raw_noise':np.c_[raw[ids],noise[ids]],
                'raw_rff':np.c_[raw[ids],rff],'raw_oracle':np.c_[raw[ids],oracle[ids]]}
            for name,clip in clips.items():
                cm=clip[ids[tr]].mean(0);_,_,vt=np.linalg.svd(clip[ids[tr]]-cm,full_matrices=False)
                values['raw_clip_'+name]=np.c_[raw[ids],(clip[ids]-cm)@vt[:16].T]
            for name,extra in extras.items():
                values[name]=np.c_[raw[ids],randomized_pca(extra[ids[tr]],extra[ids])]
            for method,x in values.items():
                pp,selection=fit_predict(x[tr],y[tr],ll[tr],x[va],y[va],ll[va],x[te]);pred[method][te]=pp
                records.append(dict(cohort=cohort,fold=f,method=method,selection=selection))
            print(cohort,'fold',f,'complete',flush=True)
        assert all(np.isfinite(v).all() for v in pred.values())
        comparisons={};unique=sorted(set(ll));draws=rng.integers(len(unique),size=(10000,len(unique)))
        for method,baseline in [('raw_fluctuation','raw'),('raw_fluctuation','raw_rff'),('raw_oracle','raw'),('raw_fluctuation','raw_noise')]+[(m,b) for m in ['raw_clip_'+k for k in clips]+list(extras) for b in ['raw','raw_noise','raw_rff']]:
            a=np.clip(pred[method],1e-8,1-1e-8);b=np.clip(pred[baseline],1e-8,1-1e-8)
            delta=-y*np.log(a/b)-(1-y)*np.log((1-a)/(1-b));perlog=np.array([delta[ll==l].mean() for l in unique])
            comparisons[method+'_minus_'+baseline]=dict(log_nll_difference=float(perlog.mean()),conditional_log_bootstrap_95=np.quantile(perlog[draws].mean(1),[.025,.975]).tolist())
        results[cohort]=dict(methods={m:metrics(y,v,ll) for m,v in pred.items()},comparisons=comparisons)
        save.update({cohort+'_ids':ids,cohort+'_labels':y,cohort+'_logs':ll,**{cohort+'_'+m:v for m,v in pred.items()}})
    np.savez_compressed(args.out/'predictions.npz',**save);(args.out/'summary.json').write_text(json.dumps(results,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(Path('docs/PROTOCOLO_TRANSICOES_E_OBSERVABILIDADE_V1.md').read_bytes()).hexdigest(),
        data=str(args.data),clip=str(args.clip),extra_features=args.extra_features,parked_stopped_only=args.parked_stopped_only,limitations=['Complete-known-outcome selected cohort; not correction for informative censoring','Prior examination of all logs possible','Frozen visual readout only; no planner training or driving evaluation','Derived fluctuation features add no information to full raw history','Human attribute may reflect retrospective annotation']),indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(results,indent=2))

if __name__=='__main__':main()
