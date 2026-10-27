"""Finite-horizon incidence-only control: p(event|x) times one shared event clock.

This is an established mixture construction, not an estimate of permanent cure.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,softmax,logsumexp
from cavla.transition_clock_probe import EDGES,person_period,path_loss,fit_hazard
from cavla.transition_history_probe import log_mean


def incidence_objective(theta,x,event,follow,weights,alpha,anchor):
    nbin=len(EDGES)-1;time=theta[:nbin];coef=theta[nbin:]
    z=x@coef;prob=expit(z);q=softmax(time);prefix=np.r_[0.,np.cumsum(q)]
    observed=np.isfinite(event)&(event>0)&(event<=EDGES[-1])
    bins=np.searchsorted(EDGES[1:],np.where(observed,event,0.),side='left')
    count=np.searchsorted(EDGES[1:],follow+1e-9,side='right').clip(max=nbin)
    mass=prefix[count];survival=(1-prob*mass).clip(min=1e-12)
    loss=np.where(observed,np.logaddexp(0.,-z)-time[bins]+logsumexp(time),-np.log(survival))
    gz=np.where(observed,prob-1,prob*(1-prob)*mass/survival)
    derivative=np.where(observed,0.,weights*prob/survival)
    gq=np.sum(derivative[:,None]*q[None,:]*((np.arange(nbin)[None,:]<count[:,None])-mass[:,None]),axis=0)
    gq+=q*np.sum(weights[observed])-np.bincount(bins[observed],weights=weights[observed],minlength=nbin)
    penalty=coef.copy();penalty[0]=0
    value=np.sum(weights*loss)+alpha*np.sum(penalty**2)/2+.5e-5*np.sum((time-anchor)**2)
    grad=np.r_[gq+1e-5*(time-anchor),x.T@(weights*gz)+alpha*penalty]
    return value,grad


def fit_incidence(train,event,follow,logs,val,ve,vf,vlogs,test):
    baseline,record=fit_hazard(np.empty((len(train),0)),event,follow,logs,np.empty((len(val),0)),ve,vf,vlogs,np.empty((len(test),0)))
    if train.shape[1]==0:return baseline,record
    _,inverse,counts=np.unique(logs,return_inverse=True,return_counts=True);weights=1/(len(counts)*counts[inverse])
    base=baseline[0];surv=np.cumprod(1-base);pmf=base*np.r_[1.,surv[:-1]];incidence=1-surv[-1]
    q=pmf/incidence;anchor=np.log(q);anchor-=anchor.mean()
    mean=train.mean(0);scale=train.std(0).clip(min=1e-6)
    transform=lambda a:np.c_[np.ones(len(a)),np.clip((a-mean)/scale,-10,10)]
    xx=transform(train);vv=transform(val);ee=transform(test)
    best=record['validation_nll'];answer=baseline;record=dict(record);record['candidates']=[]
    def predict(theta,values):
        prob=expit(values@theta[len(base):]);clock=softmax(theta[:len(base)])
        mass=prob[:,None]*clock;remaining=1-np.cumsum(mass,axis=1)
        return (mass/np.c_[np.ones(len(values)),remaining[:,:-1]].clip(min=1e-12)).clip(1e-8,1-1e-8)
    for alpha in [.001,.01,.1,1.,10.]:
        start=np.r_[anchor,np.log(incidence/(1-incidence)),np.zeros(train.shape[1])]
        # Diagonal scaling addresses the very different rarity/time and ridge curvature.
        scaling=np.r_[1/np.sqrt((incidence*q).clip(min=1e-5)),1/np.sqrt(max(incidence*(1-incidence),1e-5)),np.full(train.shape[1],1/np.sqrt(alpha+.02))]
        def objective(u):
            loss,grad=incidence_objective(u*scaling,xx,event,follow,weights,alpha,anchor)
            return loss,grad*scaling
        opt=minimize(objective,start/scaling,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-11,'gtol':1e-7})
        theta=opt.x*scaling;hz=predict(theta,vv);score=log_mean(path_loss(ve,vf,hz),vlogs)
        record['candidates'].append({'alpha':alpha,'validation_nll':score,'converged':bool(opt.success),'iterations':int(opt.nit),'message':str(opt.message)})
        if score<best:
            best=score;answer=predict(theta,ee);record.update(model='incidence_only_shared_clock',alpha=alpha,validation_nll=score)
    return answer,record


def main():
    # Reuse the exact cohort, frozen inputs, folds, metrics and serialization.
    import sys,json,hashlib
    from pathlib import Path
    snapshot=Path(__file__).read_bytes()
    output=Path(sys.argv[sys.argv.index('--out')+1])
    import cavla.transition_clock_probe as driver
    driver.fit_hazard=fit_incidence
    driver.main()
    (output/'source_snapshot_incidence.py').write_bytes(snapshot)
    manifest=json.loads((output/'manifest.json').read_text())
    manifest.update(family='finite_horizon_incidence_times_shared_clock',incidence_source_sha256=hashlib.sha256(snapshot).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
