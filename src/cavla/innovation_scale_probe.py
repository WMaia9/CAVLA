"""Exploratory frozen-CV uncertainty-scale probe with nested log splits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from cavla.temporal_innovation_probe import standardize, distance, log_mean

STEPS = np.array([9,19,39,63])

def gaussian_nll(error_energy, variance):
    """2D zero-mean isotropic Gaussian, energy = squared norm / 2."""
    variance=np.maximum(variance,.01)
    return np.log(2*np.pi*variance)+error_energy/variance


def mixture_nll(energy, variances):
    return -logsumexp(-np.stack([gaussian_nll(energy,v) for v in variances]),axis=0)+np.log(len(variances))


def select_scale(family, train, energy, val, val_energy, evaluation, val_logs):
    baseline=np.maximum(energy.mean(0),.01)
    best=log_mean(gaussian_nll(val_energy,baseline).mean(1),val_logs)
    predictions=[np.broadcast_to(baseline,(len(e),len(baseline))).copy() for e in evaluation]
    selection=dict(correction='global_variance',validation_log_nll=best)
    target=np.log(.01+energy); candidates=[]
    alphas=[.01,.1,1.,10.,100.,1000.] if family=='linear' else [.001,.01,.1,1.,10.]
    for bandwidth in ([None] if family=='linear' else [.25,1.,4.]):
        if family=='linear':
            tx=np.c_[np.ones(len(train)),train]; vx=np.c_[np.ones(len(val)),val]
            ex=[np.c_[np.ones(len(e)),e] for e in evaluation]
            matrix=tx.T@tx; rhs=tx.T@target
        else:
            tx=np.exp(-distance(train,train)/bandwidth);vx=np.exp(-distance(val,train)/bandwidth)
            ex=[np.exp(-distance(e,train)/bandwidth) for e in evaluation];matrix=tx;rhs=target
        eig,basis=np.linalg.eigh(matrix)
        for alpha in alphas:
            coefficient=basis@((basis.T@rhs)/(np.maximum(eig,0)[:,None]+alpha))
            smearing=np.exp(np.clip(target-tx@coefficient,-30,30)).mean(0)
            predict=lambda a:np.maximum(.01,np.exp(np.clip(a@coefficient,-30,30))*smearing-.01)
            variance=predict(vx)
            score=log_mean(gaussian_nll(val_energy,variance).mean(1),val_logs)
            candidates.append(dict(alpha=alpha,bandwidth=bandwidth,validation_log_nll=score))
            if score<best:
                best=score;predictions=[predict(e) for e in ex]
                selection=dict(correction=family,alpha=alpha,bandwidth=bandwidth,validation_log_nll=best)
    selection['candidates']=candidates
    return predictions,selection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    loaded=[];times=(STEPS+1)*.1
    for tag,feature in [('development','artifacts/object_features_20261021/features.npz'),
                        ('previously_used_holdout','artifacts/composition_holdout_features_v1_20261021/features.npz')]:
        with np.load(f'artifacts/temporal_necessity_v2_20261021/{tag}_labels_predictions.npz') as z:
            use=z['eligible_past'];d={k:z[k][use] for k in ['numeric','future_centers','object_indices','logs','scenes']}
        cv=d['numeric'][:,None,10:12]+times[None,:,None]*d['numeric'][:,None,12:14]
        d['energy']=np.square(d['future_centers'][:,STEPS]-cv).sum(-1)/2
        with np.load(feature) as z:d['visual']=z['features'][d['object_indices']].reshape(len(d['numeric']),-1).astype(float)
        loaded.append(d)
    dev,hold=loaded;rng=np.random.default_rng(20261021)
    noise=[rng.normal(size=(len(d['numeric']),16)) for d in loaded]
    methods=['global_variance']+[f'{fam}_{feat}' for fam in ['linear','rbf'] for feat in ['numeric','vla','noise']]
    oof={m:np.full_like(dev['energy'],np.nan) for m in methods}
    hold_vars={m:[] for m in methods};records=[]
    for fold in range(3):
        split=json.loads(Path(f'artifacts/structured_geometry_log_validation_20261021/fold{fold}_split.json').read_text())
        ti,vi,ei=[np.flatnonzero(np.isin(dev['object_indices'],split[name+'_objects'])) for name in ['train','validation','test']]
        assert not(set(dev['logs'][ti])&(set(dev['logs'][vi])|set(dev['logs'][ei])|set(hold['logs'])))
        numeric=standardize(dev['numeric'][ti],dev['numeric'][vi],dev['numeric'][ei],hold['numeric'])
        mean=dev['visual'][ti].mean(0);_,_,vt=np.linalg.svd(dev['visual'][ti]-mean,full_matrices=False)
        visual=standardize(*[(a-mean)@vt[:16].T for a in [dev['visual'][ti],dev['visual'][vi],dev['visual'][ei],hold['visual']]])
        random=standardize(noise[0][ti],noise[0][vi],noise[0][ei],noise[1])
        for method in methods:
            if method=='global_variance':
                base=np.maximum(dev['energy'][ti].mean(0),.01)
                pred=[np.broadcast_to(base,(len(ids),4)).copy() for ids in [ei,hold['logs']]]
                selected=dict(correction='global_variance')
            else:
                family,features=method.split('_')
                values=numeric if features=='numeric' else [np.c_[a,b] for a,b in zip(numeric,visual if features=='vla' else random)]
                pred,selected=select_scale(family,values[0],dev['energy'][ti],values[1],dev['energy'][vi],values[2:],dev['logs'][vi])
            assert np.isfinite(pred[0]).all() and np.isfinite(pred[1]).all()
            oof[method][ei]=gaussian_nll(dev['energy'][ei],pred[0]);hold_vars[method].append(pred[1])
            records.append(dict(fold=fold,method=method,selection=selected))
        print('fold',fold,'complete',flush=True)
    losses={m:mixture_nll(hold['energy'],hold_vars[m]) for m in methods}
    assert all(np.isfinite(v).all() for v in oof.values())
    unique=sorted(set(hold['logs']));draws=rng.integers(len(unique),size=(10000,len(unique)))
    comparisons={}
    for family in ['linear','rbf']:
        for baseline in ['numeric','noise']:
            delta=(losses[family+'_vla']-losses[family+'_'+baseline]).mean(1)
            perlog=np.array([delta[hold['logs']==l].mean() for l in unique])
            comparisons[family+'_vla_minus_'+baseline]=dict(difference_log_mean_nll=float(perlog.mean()),
                conditional_log_bootstrap_95=np.quantile(perlog[draws].mean(1),[.025,.975]).tolist(),
                log_differences=dict(zip(unique,perlog.tolist())))
    summary=dict(status='Exploratory information about error scale; no novel method or driving claim',
        development_agents=len(dev['numeric']),evaluation_agents=len(hold['numeric']),evaluation_logs=len(unique),horizons_seconds=times.tolist(),
        cv_log_mean_nll={m:log_mean(v.mean(1),dev['logs']) for m,v in oof.items()},
        evaluation_log_mean_nll={m:log_mean(v.mean(1),hold['logs']) for m,v in losses.items()},
        evaluation_log_nll_by_horizon={m:[log_mean(v[:,j],hold['logs']) for j in range(4)] for m,v in losses.items()},
        comparisons=comparisons,limitations=['Previously used evaluation','Fixed-model conditional bootstrap','Gaussian scale probe, not full distribution',
        'Past privileged 3D state','No calibrated coverage or decision gain','Only current frontal ROI PCA16 features'])
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(args.out/'predictions.npz',development_logs=dev['logs'],evaluation_logs=hold['logs'],evaluation_energy=hold['energy'],
        **{'cv_nll_'+m:v for m,v in oof.items()},**{'evaluation_variances_'+m:np.stack(v) for m,v in hold_vars.items()},**{'evaluation_nll_'+m:v for m,v in losses.items()})
    (args.out/'manifest.json').write_text(json.dumps(dict(seed=20261021,protocol_sha256=hashlib.sha256(Path('docs/PROTOCOLO_FISICA_INFORMACAO_V1.md').read_bytes()).hexdigest()),indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
