"""Post-hoc unconditional heavy-tail controls for the visual scale pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from cavla.innovation_scale_probe import STEPS, gaussian_nll, mixture_nll
from cavla.temporal_innovation_probe import log_mean


def component_loss(energy, variances, weights):
    return -logsumexp(-gaussian_nll(energy[:,None,:],variances[None,:,:])+np.log(weights)[None,:,:],axis=1)


def fit_mixture(energy, logs, components, steps=200):
    """EM: 2D zero-mean isotropic components, separate weights per horizon."""
    sample_weight=np.array([1/(len(set(logs))*(logs==l).sum()) for l in logs])
    variance=np.maximum(.01,np.quantile(energy,np.linspace(.05,.95,components),axis=0))
    weights=np.full_like(variance,1/components);history=[]
    for _ in range(steps):
        logprob=-gaussian_nll(energy[:,None,:],variance[None,:,:])+np.log(np.maximum(weights,1e-15))[None,:,:]
        history.append(float(np.sum(-logsumexp(logprob,axis=1)*sample_weight[:,None])))
        responsibility=np.exp(logprob-logsumexp(logprob,axis=1,keepdims=True))*sample_weight[:,None,None]
        mass=responsibility.sum(0)
        weights=mass/mass.sum(0)
        variance=np.maximum(.01,(responsibility*energy[:,None,:]).sum(0)/np.maximum(mass,1e-15))
    return variance,weights,history


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    base=Path('artifacts/innovation_scale_v1_20261021')
    with np.load('artifacts/temporal_necessity_v2_20261021/development_labels_predictions.npz') as z:
        u=z['eligible_past'];n=z['numeric'][u];ids=z['object_indices'][u];logs=z['logs'][u];future=z['future_centers'][u]
    time=(STEPS+1)*.1;cv=n[:,None,10:12]+time[None,:,None]*n[:,None,12:14]
    energy=np.square(future[:,STEPS]-cv).sum(-1)/2
    with np.load(base/'predictions.npz') as z:
        he=z['evaluation_energy'];hl=z['evaluation_logs']
        vla={f:z['evaluation_nll_'+f+'_vla'] for f in ['linear','rbf']}
    controlvars={k:[] for k in ['train_log_weighted','validation_log_weighted','train_validation_log_weighted']}
    records=[];selected_losses=[];oof=np.full_like(energy,np.nan)
    for fold in range(3):
        s=json.loads(Path(f'artifacts/structured_geometry_log_validation_20261021/fold{fold}_split.json').read_text())
        ti,vi,ei=[np.isin(ids,s[key+'_objects']) for key in ['train','validation','test']]
        for name,mask in zip(controlvars,[ti,vi,ti|vi]):
            var=np.maximum(.01,[log_mean(energy[mask,j],logs[mask]) for j in range(4)])
            controlvars[name].append(np.broadcast_to(var,he.shape))
        best=np.inf;record={'fold':fold,'candidates':[]}
        for k in [1,2,4,8]:
            variance,weights,history=fit_mixture(energy[ti],logs[ti],k)
            assert np.max(np.diff(history))<1e-7
            score=log_mean(component_loss(energy[vi],variance,weights).mean(1),logs[vi])
            record['candidates'].append(dict(components=k,validation_nll=score,variance=variance.tolist(),weights=weights.tolist()))
            if score<best:
                best=score;record['selected_components']=k
                h_loss=component_loss(he,variance,weights);e_loss=component_loss(energy[ei],variance,weights)
        selected_losses.append(h_loss);oof[ei]=e_loss;records.append(record)
    losses={name:mixture_nll(he,v) for name,v in controlvars.items()}
    losses['unconditional_scale_mixture']=-logsumexp(-np.stack(selected_losses),axis=0)+np.log(3)
    output=dict(status='POST-HOC stronger baselines after viewing v1 results; not independent confirmation',
                cv_log_nll_unconditional_mixture=log_mean(oof.mean(1),logs),controls={})
    rng=np.random.default_rng(20261021);unique=sorted(set(hl));draws=rng.integers(len(unique),size=(10000,len(unique)))
    for name,loss in losses.items():
        comparisons={}
        for family,l in vla.items():
            a=np.array([(l-loss)[hl==key].mean() for key in unique])
            comparisons[family+'_vla_minus_control']=dict(mean=float(a.mean()),conditional_log_bootstrap_95=np.quantile(a[draws].mean(1),[.025,.975]).tolist(),
                    leave_one_log_out_range=[float((a.sum()-x)/(len(a)-1)) for x in [a.max(),a.min()]])
        output['controls'][name]=dict(log_mean_nll=log_mean(loss.mean(1),hl),comparisons=comparisons)
    (args.out/'summary.json').write_text(json.dumps(output,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(args.out/'losses.npz',**losses,evaluation_logs=hl,cv_mixture_nll=oof)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(output,indent=2))

if __name__=='__main__':main()
