"""Strong nonlinear numeric-only discrete hazard control, with log-held-out selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cavla.transition_clock_probe import EDGES,person_period,path_loss,fit_hazard
from cavla.transition_history_probe import log_mean,metrics


def mix_hazards(base,hazard,weight):
    """Mixture of path/event laws, rather than independent per-period mixtures."""
    def mass(h):return h*np.c_[np.ones(len(h)),np.cumprod(1-h,axis=1)[:,:-1]]
    pmf=(1-weight)*mass(base)+weight*mass(hazard)
    remaining=np.c_[np.ones(len(pmf)),1-np.cumsum(pmf,axis=1)[:,:-1]]
    return (pmf/remaining.clip(min=1e-12)).clip(1e-8,1-1e-8)


def fit_tree(train,event,follow,logs,val,ve,vf,vlogs,test):
    from sklearn.ensemble import HistGradientBoostingClassifier
    hz,record=fit_hazard(np.empty((len(train),0)),event,follow,logs,np.empty((len(val),0)),ve,vf,vlogs,np.empty((len(test),0)))
    best=record['validation_nll'];record=dict(record);record['candidates']=[]
    base_test=hz.copy();base_val=np.tile(hz[0],(len(val),1))
    rows,bins,y=person_period(event,follow);xx=np.c_[train[rows],bins]
    _,inverse,counts=np.unique(logs,return_inverse=True,return_counts=True);weights=len(train)/(len(counts)*counts[inverse])
    def predict(model,values):
        nbin=len(EDGES)-1;period=np.tile(np.arange(nbin),len(values));expanded=np.repeat(values,nbin,axis=0)
        return model.predict_proba(np.c_[expanded,period])[:,1].reshape(len(values),nbin)
    for leaves in [7,15,31]:
        for minimum in [20,100]:
            model=HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=leaves,min_samples_leaf=minimum,
                l2_regularization=1.,max_bins=127,early_stopping=False,random_state=20261021)
            model.fit(xx,y.astype(int),sample_weight=weights[rows]);vh=predict(model,val);th=None
            for blend in [.1,.25,.5,.75,1.]:
                calibrated=mix_hazards(base_val,vh,blend);score=log_mean(path_loss(ve,vf,calibrated),vlogs)
                record['candidates'].append({'max_leaf_nodes':leaves,'min_samples_leaf':minimum,'mixture_weight':blend,'validation_nll':score})
                if score<best:
                    if th is None:th=predict(model,test)
                    best=score;hz=mix_hazards(base_test,th,blend);record.update(model='hist_gradient_boosting_mixture',max_leaf_nodes=leaves,min_samples_leaf=minimum,mixture_weight=blend,validation_nll=score)
    return hz,record


def main():
    import sklearn
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    objects=args.data/'objects.json';rows=json.loads(objects.read_text())
    with np.load(args.data/'features.npz') as z:arrays={k:z[k] for k in z.files}
    cats=sorted({r['category'] for r in rows});fixed=np.array([[r['category']==c for c in cats]+r['size'] for r in rows],float)
    numeric=np.c_[fixed,arrays['basic'],arrays['relative_geometry'],arrays['raw_history']]
    logs=np.array([r['log'] for r in rows]);folds=np.array([int(hashlib.sha256(l.encode()).hexdigest(),16)%5 for l in logs])
    event=np.array([np.nan if r['event_seconds'] is None else r['event_seconds'] for r in rows]);follow=np.array([r['followup_seconds'] for r in rows]);strict=np.array([r['strict_stationary'] for r in rows])
    records=[];summary={};save={}
    for cohort,mask in [('principal',np.ones(len(rows),bool)),('strict',strict)]:
        ids=np.flatnonzero(mask);ff=folds[ids];ll=logs[ids];ev=event[ids];fo=follow[ids];x=numeric[ids];hz=np.full((len(ids),len(EDGES)-1),np.nan)
        for f in range(5):
            tr=np.flatnonzero((ff!=f)&(ff!=(f+1)%5));va=np.flatnonzero(ff==(f+1)%5);te=np.flatnonzero(ff==f)
            assert not(set(ll[tr])&(set(ll[va])|set(ll[te])))
            pred,record=fit_tree(x[tr],ev[tr],fo[tr],ll[tr],x[va],ev[va],fo[va],ll[va],x[te]);hz[te]=pred
            records.append({'cohort':cohort,'fold':f,'selection':record});print(cohort,'fold',f,'complete',flush=True)
        assert np.isfinite(hz).all();result={'observed_path_log_nll':log_mean(path_loss(ev,fo,hz),ll),'horizons':{}}
        for h in [2.,4.,6.4]:
            y=np.isfinite(ev)&(ev<=h);known=y|(fo>=h);b=int(np.searchsorted(EDGES,h))-1;risk=1-np.prod(1-hz[:,:b+1],axis=1)
            result['horizons'][str(h)]=metrics(y[known].astype(float),risk[known],ll[known])
        summary[cohort]=result;save.update({cohort+'_ids':ids,cohort+'_logs':ll,cohort+'_numeric_tree':hz})
    np.savez_compressed(args.out/'predictions.npz',**save);(args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'data':str(args.data),'sklearn_version':sklearn.__version__,
        'objects_sha256':hashlib.sha256(objects.read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((args.data/'features.npz').read_bytes()).hexdigest(),
        'scope':'Established nonlinear numeric control, same log folds and observed path score; no new method or policy evaluation'},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
