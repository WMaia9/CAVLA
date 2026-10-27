"""Post-hoc matched-family information probe, not a novel mixture method."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from cavla.innovation_scale_probe import STEPS, gaussian_nll
from cavla.temporal_innovation_probe import standardize,log_mean


def gate_loss_grad(flat,x,log_density,offset,sample_weight,alpha):
    w=flat.reshape(x.shape[1],*offset.shape)
    logits=np.einsum('nd,dhk->nhk',x,w)+offset[None]
    logweight=logits-logsumexp(logits,axis=-1,keepdims=True)
    joint=logweight+log_density
    responsibility=np.exp(joint-logsumexp(joint,axis=-1,keepdims=True))
    objective=-np.sum(sample_weight[:,None]*logsumexp(joint,axis=-1))/offset.shape[0]+alpha*np.square(w).sum()/2
    deriv=(np.exp(logweight)-responsibility)*sample_weight[:,None,None]/offset.shape[0]
    grad=np.einsum('nd,nhk->dhk',x,deriv)+alpha*w
    return float(objective),grad.reshape(-1)


def predict_nll(x,energy,variance,weights,w=None):
    offset=np.log(np.maximum(weights.T,1e-15))
    logits=np.broadcast_to(offset,(len(x),*offset.shape)).copy()
    if w is not None:logits+=np.einsum('nd,dhk->nhk',x,w)
    lw=logits-logsumexp(logits,axis=-1,keepdims=True)
    density=-gaussian_nll(energy[:,:,None],variance.T[None])
    return -logsumexp(lw+density,axis=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    loaded=[]
    for tag,fp in [('development','artifacts/object_features_20261021/features.npz'),('previously_used_holdout','artifacts/composition_holdout_features_v1_20261021/features.npz')]:
        with np.load(f'artifacts/temporal_necessity_v2_20261021/{tag}_labels_predictions.npz') as z:
            use=z['eligible_past'];d={k:z[k][use] for k in ['numeric','future_centers','logs','object_indices']}
        cv=d['numeric'][:,None,10:12]+((STEPS+1)*.1)[None,:,None]*d['numeric'][:,None,12:14]
        d['energy']=np.square(d['future_centers'][:,STEPS]-cv).sum(-1)/2
        with np.load(fp) as z:d['visual']=z['features'][d['object_indices']].reshape(len(d['numeric']),-1).astype(float)
        loaded.append(d)
    dev,hold=loaded;rng=np.random.default_rng(20261021)
    noise=[rng.normal(size=(len(d['numeric']),16)) for d in loaded]
    controls=json.loads(Path('artifacts/innovation_scale_strong_controls_v1_20261021/records.json').read_text())
    methods=['unconditional','numeric','vla','noise'];oof={m:np.full_like(dev['energy'],np.nan) for m in methods};hloss={m:[] for m in methods};records=[]
    for fold in range(3):
        s=json.loads(Path(f'artifacts/structured_geometry_log_validation_20261021/fold{fold}_split.json').read_text())
        ti,vi,ei=[np.flatnonzero(np.isin(dev['object_indices'],s[key+'_objects'])) for key in ['train','validation','test']]
        assert not(set(dev['logs'][ti])&(set(dev['logs'][vi])|set(dev['logs'][ei])|set(hold['logs'])))
        selected=next(c for c in controls[fold]['candidates'] if c['components']==controls[fold]['selected_components'])
        variance=np.array(selected['variance']);weights=np.array(selected['weights']);offset=np.log(np.maximum(weights.T,1e-15))
        base=standardize(dev['numeric'][ti],dev['numeric'][vi],dev['numeric'][ei],hold['numeric'])
        mean=dev['visual'][ti].mean(0);_,_,vt=np.linalg.svd(dev['visual'][ti]-mean,full_matrices=False)
        visual=standardize(*[(a-mean)@vt[:16].T for a in [dev['visual'][ti],dev['visual'][vi],dev['visual'][ei],hold['visual']]])
        random=standardize(noise[0][ti],noise[0][vi],noise[0][ei],noise[1])
        sample_weight=np.array([1/(len(set(dev['logs'][ti]))*(dev['logs'][ti]==l).sum()) for l in dev['logs'][ti]])
        density=-gaussian_nll(dev['energy'][ti,:,None],variance.T[None])
        for method in methods:
            feats=base if method in ['unconditional','numeric'] else [np.c_[a,b] for a,b in zip(base,visual if method=='vla' else random)]
            x=[np.c_[np.ones(len(a)),a] for a in feats]
            best=log_mean(predict_nll(x[1],dev['energy'][vi],variance,weights).mean(1),dev['logs'][vi]);winner=None
            selection=dict(model='unconditional',validation_nll=best,candidates=[])
            if method!='unconditional':
                for alpha in [.001,.01,.1,1.]:
                    opt=minimize(gate_loss_grad,np.zeros(x[0].shape[1]*offset.size),args=(x[0],density,offset,sample_weight,alpha),jac=True,method='L-BFGS-B',options=dict(maxiter=300,ftol=1e-10))
                    w=opt.x.reshape(x[0].shape[1],*offset.shape)
                    score=log_mean(predict_nll(x[1],dev['energy'][vi],variance,weights,w).mean(1),dev['logs'][vi])
                    selection['candidates'].append(dict(alpha=alpha,validation_nll=score,converged=bool(opt.success),iterations=int(opt.nit)))
                    if score<best:best=score;winner=w;selection.update(model='conditional',alpha=alpha,validation_nll=score)
            oof[method][ei]=predict_nll(x[2],dev['energy'][ei],variance,weights,winner)
            hloss[method].append(predict_nll(x[3],hold['energy'],variance,weights,winner))
            records.append(dict(fold=fold,method=method,components=len(variance),selection=selection))
        print('fold',fold,'complete',flush=True)
    losses={m:-logsumexp(-np.stack(v),axis=0)+np.log(3) for m,v in hloss.items()}
    comparisons={};unique=sorted(set(hold['logs']));draws=rng.integers(len(unique),size=(10000,len(unique)))
    for baseline in ['unconditional','numeric','noise']:
        a=np.array([(losses['vla']-losses[baseline])[hold['logs']==l].mean() for l in unique])
        comparisons['vla_minus_'+baseline]=dict(mean=float(a.mean()),conditional_log_bootstrap_95=np.quantile(a[draws].mean(1),[.025,.975]).tolist(),logs_improved=int((a<0).sum()))
    summary=dict(status='Adaptive post-hoc matched-family scale-mixture pilot; no novel method',development_agents=len(dev['logs']),evaluation_agents=len(hold['logs']),evaluation_logs=len(unique),
        cv_log_nll={m:log_mean(v.mean(1),dev['logs']) for m,v in oof.items()},evaluation_log_nll={m:log_mean(v.mean(1),hold['logs']) for m,v in losses.items()},comparisons=comparisons)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(args.out/'losses.npz',development_logs=dev['logs'],evaluation_logs=hold['logs'],**{'cv_'+m:v for m,v in oof.items()},**{'evaluation_'+m:v for m,v in losses.items()})
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
