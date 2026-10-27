"""Established discrete-hazard baselines as a temporal-information feasibility gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from cavla.transition_history_probe import randomized_pca,metrics,log_mean

EDGES=np.r_[np.arange(0.,6.1,.5),6.4]


def person_period(event,follow):
    row=[];period=[];labels=[]
    for i,(e,c) in enumerate(zip(event,follow)):
        is_event=np.isfinite(e) and 0<e<=EDGES[-1]
        stop=int(np.searchsorted(EDGES[1:],e,side='left')) if is_event else int(np.searchsorted(EDGES[1:],c+1e-9,side='right'))-1
        for b in range(stop+1):
            row.append(i);period.append(b);labels.append(float(is_event and b==stop))
    return np.array(row,int),np.array(period,int),np.array(labels,float)


def path_loss(event,follow,hazard):
    rows,bins,labels=person_period(event,follow)
    hp=np.clip(hazard[rows,bins],1e-8,1-1e-8)
    losses=-labels*np.log(hp)-(1-labels)*np.log1p(-hp)
    return np.bincount(rows,weights=losses,minlength=len(event))


def fit_hazard(train,event,follow,logs,val,ve,vf,vlogs,test):
    nbin=len(EDGES)-1;rows,bins,labels=person_period(event,follow)
    unique,inverse,counts=np.unique(logs,return_inverse=True,return_counts=True)
    weights=1/(len(unique)*counts[inverse]);w=weights[rows]
    total=np.bincount(bins,weights=w,minlength=nbin)
    hits=np.bincount(bins,weights=w*labels,minlength=nbin)
    # Pseudocount 0.5 examples per log-weighted bin; equivalent average-actor weight.
    smooth=.5/len(train);base=np.clip((hits+smooth)/(total+2*smooth),1e-6,1-1e-6)
    baseline=np.tile(base,(len(val),1));best=log_mean(path_loss(ve,vf,baseline),vlogs)
    answer=np.tile(base,(len(test),1));record={'model':'bin_hazard','validation_nll':best,'candidates':[]}
    if train.shape[1]==0:return answer,record
    mean=train.mean(0);scale=train.std(0).clip(min=1e-6)
    xx=np.clip((train-mean)/scale,-10,10);vv=np.clip((val-mean)/scale,-10,10);ee=np.clip((test-mean)/scale,-10,10)
    for alpha in [.001,.01,.1,1.,10.]:
        curvature=w*base[bins]*(1-base[bins])
        diagonal=np.r_[np.bincount(bins,weights=curvature,minlength=nbin)+1e-5,
            (xx[rows]**2).T@curvature+alpha]
        parameter_scale=1/np.sqrt(diagonal.clip(min=1e-8))
        def objective(theta):
            theta=theta*parameter_scale
            time=theta[:nbin];coef=theta[nbin:];z=time[bins]+xx[rows]@coef
            residual=w*(expit(z)-labels)
            value=np.sum(w*(np.logaddexp(0,z)-labels*z))+alpha*np.sum(coef**2)/2
            # Tiny quadratic anchor prevents empty bins from drifting without data.
            anchor=1e-5*(time-np.log(base/(1-base)))
            value+=.5e-5*np.sum((time-np.log(base/(1-base)))**2)
            gradient=np.r_[np.bincount(bins,weights=residual,minlength=nbin)+anchor,xx[rows].T@residual+alpha*coef]
            return value,gradient*parameter_scale
        start=np.r_[np.log(base/(1-base)),np.zeros(train.shape[1])]
        opt=minimize(objective,start/parameter_scale,jac=True,method='L-BFGS-B',options={'maxiter':700,'ftol':1e-11,'gtol':1e-7})
        theta=opt.x*parameter_scale
        hz=expit(theta[:nbin]+vv@theta[nbin:,None]);score=log_mean(path_loss(ve,vf,hz),vlogs)
        record['candidates'].append({'alpha':alpha,'validation_nll':score,'converged':bool(opt.success),'iterations':int(opt.nit),'message':str(opt.message)})
        if score<best:
            best=score;answer=expit(theta[:nbin]+ee@theta[nbin:,None]);record.update(model='discrete_hazard',alpha=alpha,validation_nll=score)
    return answer,record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('artifacts/front_transition_data_v1_20261021'))
    p.add_argument('--extra-features',action='append',default=[],help='NAME=DIRECTORY, last-layer ROI and ROI/context PCA16')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=args.data;objects=source/'objects.json';rows=json.loads(objects.read_text())
    sha=hashlib.sha256(objects.read_bytes()).hexdigest()
    with np.load(source/'features.npz') as z:arrays={k:z[k] for k in z.files}
    logs=np.array([r['log'] for r in rows]);folds=np.array([int(hashlib.sha256(l.encode()).hexdigest(),16)%5 for l in logs])
    event=np.array([np.nan if r['event_seconds'] is None else r['event_seconds'] for r in rows]);follow=np.array([r['followup_seconds'] for r in rows])
    cats=sorted({r['category'] for r in rows});attrs=sorted({r['attribute'] for r in rows})
    fixed=np.array([[r['category']==c for c in cats]+r['size'] for r in rows],float)
    basic=np.c_[fixed,arrays['basic'],arrays['relative_geometry']];raw=np.c_[basic,arrays['raw_history']]
    oracle=np.array([[r['attribute']==a for a in attrs] for r in rows],float)
    rng=np.random.default_rng(20261021);noise=rng.normal(size=(len(rows),16))
    visual={}
    for name,directory in [('clip','front_transition_clip_v1_20261021'),('r1','front_transition_r1_v1_20261021'),('qwen','front_transition_qwen_v1_20261021')]:
        path=Path('artifacts')/directory;manifest=json.loads((path/'manifest.json').read_text());assert manifest['objects_sha256']==sha
        with np.load(path/'features.npz') as z:
            if name=='clip':visual['raw_clip_both']=z['features'].astype(float)
            else:
                roi=z['features'][:,-1].astype(float);ctx=z['context'][:,-1].astype(float)
                if name=='r1':visual['raw_r1_roi']=roi
                visual['raw_'+name+'_both']=np.c_[roi,ctx]
    for spec in args.extra_features:
        name,directory=spec.split('=',1);path=Path(directory);manifest=json.loads((path/'manifest.json').read_text());assert manifest['objects_sha256']==sha
        with np.load(path/'features.npz') as z:
            assert int(z['layers'][-1])==36
            roi=z['features'][:,-1].astype(float);ctx=z['context'][:,-1].astype(float)
        for suffix,values in [('roi',roi),('both',np.c_[roi,ctx])]:
            key='raw_'+name+'_'+suffix;assert key not in visual;assert len(values)==len(rows) and np.isfinite(values).all();visual[key]=values
    strict=np.array([r['strict_stationary'] for r in rows]);summary={};records=[];save={}
    for cohort,mask in [('principal',np.ones(len(rows),bool)),('strict',strict)]:
        ids=np.flatnonzero(mask);ff=folds[ids];ll=logs[ids];ev=event[ids];fo=follow[ids]
        methods=['bin_hazard','basic','raw','raw_noise','raw_rff','raw_oracle']+list(visual)
        predictions={m:np.full((len(ids),len(EDGES)-1),np.nan) for m in methods}
        for f in range(5):
            tr=np.flatnonzero((ff!=f)&(ff!=(f+1)%5));va=np.flatnonzero(ff==(f+1)%5);te=np.flatnonzero(ff==f)
            assert not(set(ll[tr])&(set(ll[va])|set(ll[te])))
            mu=raw[ids[tr]].mean(0);scale=raw[ids[tr]].std(0).clip(min=1e-6)
            rrng=np.random.default_rng(20261021);w=rrng.normal(size=(raw.shape[1],64))/np.sqrt(raw.shape[1]);phase=rrng.uniform(0,2*np.pi,64)
            rff=np.sqrt(2/64)*np.cos(np.clip((raw[ids]-mu)/scale,-10,10)@w+phase)
            values={'bin_hazard':np.empty((len(ids),0)),'basic':basic[ids],'raw':raw[ids],
                'raw_noise':np.c_[raw[ids],noise[ids]],'raw_rff':np.c_[raw[ids],rff],'raw_oracle':np.c_[raw[ids],oracle[ids]]}
            for name,features in visual.items():values[name]=np.c_[raw[ids],randomized_pca(features[ids[tr]],features[ids])]
            for name,x in values.items():
                pred,record=fit_hazard(x[tr],ev[tr],fo[tr],ll[tr],x[va],ev[va],fo[va],ll[va],x[te])
                predictions[name][te]=pred;records.append(dict(cohort=cohort,fold=f,method=name,selection=record))
            print(cohort,'fold',f,'complete',flush=True)
        assert all(np.isfinite(h).all() for h in predictions.values())
        losses={name:path_loss(ev,fo,hz) for name,hz in predictions.items()};results={}
        for name,hz in predictions.items():
            result={'observed_path_log_nll':log_mean(losses[name],ll),'horizons':{}}
            for horizon in [2.,4.,6.4]:
                y=np.isfinite(ev)&(ev<=horizon);known=y|(fo>=horizon);b=int(np.searchsorted(EDGES,horizon))-1
                risk=1-np.prod(1-hz[:,:b+1],axis=1)
                result['horizons'][str(horizon)]=metrics(y[known].astype(float),risk[known],ll[known])
            results[name]=result
        unique=sorted(set(ll));draws=rng.integers(len(unique),size=(10000,len(unique)));comparisons={}
        pairs=[(name,baseline) for name in ['raw_oracle']+list(visual) for baseline in ['raw','raw_noise','raw_rff','bin_hazard']]
        for suffix in ['roi','both']:
            if 'raw_temporal_'+suffix in losses and 'raw_repeat_'+suffix in losses:
                pairs.extend([('raw_temporal_'+suffix,'raw_repeat_'+suffix),('raw_temporal_'+suffix,'raw_qwen_both'),('raw_temporal_'+suffix,'raw_clip_both')])
        for name,baseline in pairs:
            delta=losses[name]-losses[baseline];perlog=np.array([delta[ll==l].mean() for l in unique])
            comparisons[name+'_minus_'+baseline]={'observed_path_log_nll_difference':float(perlog.mean()),
                'conditional_log_bootstrap_95':np.quantile(perlog[draws].mean(1),[.025,.975]).tolist()}
        summary[cohort]={'agents':len(ids),'events':int(np.isfinite(ev).sum()),'methods':results,'comparisons':comparisons}
        save.update({cohort+'_ids':ids,cohort+'_logs':ll,**{cohort+'_'+m:h for m,h in predictions.items()}})
    np.savez_compressed(args.out/'predictions.npz',**save)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'objects_sha256':sha,'edges':EDGES.tolist(),
        'data':str(args.data),'extra_features':args.extra_features,
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_TRANSICOES_E_OBSERVABILIDADE_V1.md').read_bytes()).hexdigest(),
        'limitations':['Existing discrete-hazard method, not a new contribution','Exploratory log folds previously examined','No correction for informative censoring','Observed likelihood only; no claim of true future or safe policy identification']},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
