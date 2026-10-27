"""Readout of annotated parked/stopped state; not a driving method or intent oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from cavla.temporal_innovation_probe import distance, standardize


def auc(y, p):
    y = np.asarray(y, bool); n = y.sum(); m = (~y).sum()
    return float((rankdata(p)[y].sum() - n*(n+1)/2) / (n*m)) if n and m else None


def threshold(y, p):
    candidates = np.r_[np.unique(p), np.nextafter(np.max(p), np.inf)]
    scores = [((p[y] < t).mean() + (p[~y] >= t).mean())/2 for t in candidates]
    return float(candidates[np.argmin(scores)])


def metrics(y, p, t):
    return {'auc': auc(y, p), 'balanced_error': float(((p[y] < t).mean() + (p[~y] >= t).mean())/2),
            'positive': int(y.sum()), 'negative': int((~y).sum())}


def within_scene_auc(y, p, scenes):
    values = [auc(y[scenes == s], p[scenes == s]) for s in sorted(set(scenes))]
    values = [v for v in values if v is not None]
    return {'mixed_scenes': len(values), 'within_scene_mean_auc': float(np.mean(values)) if values else None}


def fit_logistic(x, y, alpha):
    x = np.c_[np.ones(len(x)), x]; y = y.astype(float)
    weights = np.where(y, .5/y.mean(), .5/(1-y.mean()))
    def objective(b):
        z = x@b
        value = np.mean(weights*(np.logaddexp(0,z) - y*z)) + alpha*np.square(b[1:]).sum()/2
        grad = x.T@(weights*(expit(z)-y))/len(y)
        grad[1:] += alpha*b[1:]
        return value, grad
    answer = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-8})
    if not answer.success and np.linalg.norm(answer.jac) > 1e-4:
        raise RuntimeError(answer.message)
    return answer.x


def choose(family, values, y, vy):
    x, v, *evaluations = values
    best = .5; selected = {'correction': 'constant'}
    vp = np.repeat(float(y.mean()), len(v)); output = [np.repeat(float(y.mean()), len(a)) for a in evaluations]
    candidates = []
    for bandwidth in ([None] if family == 'logistic' else [.25, 1., 4.]):
        matrix = np.exp(-distance(x, x)/bandwidth) if family == 'rbf' else None
        if matrix is not None:
            eig, basis = np.linalg.eigh(matrix)
        for alpha in ([.001, .01, .1, 1., 10.] if family == 'logistic' else [.001, .01, .1, 1., 10.]):
            if family == 'logistic':
                beta = fit_logistic(x, y, alpha)
                predict = lambda a: expit(np.c_[np.ones(len(a)), a]@beta)
            else:
                beta = basis@((basis.T@y)/(eig.clip(min=0)+alpha))
                predict = lambda a: np.exp(-distance(a, x)/bandwidth)@beta
            candidate = predict(v); score = auc(vy, candidate)
            candidates.append({'alpha': alpha, 'bandwidth': bandwidth, 'validation_auc': score})
            if score > best:
                best = score; vp = candidate; output = [predict(a) for a in evaluations]
                selected = {'correction': family, 'alpha': alpha, 'bandwidth': bandwidth}
    selected.update(validation_auc=best, candidates=candidates)
    return output, threshold(vy, vp), selected


def bootstrap_auc(y, p, logs, draws):
    names = sorted(set(logs)); ids = np.array([names.index(log) for log in logs])
    weights = np.zeros((len(draws), len(names)))
    np.add.at(weights, (np.arange(len(draws))[:,None], draws), 1)
    pos = weights[:, ids[y]]; neg = weights[:, ids[~y]]
    kernel = (p[y, None] > p[None, ~y]) + .5*(p[y, None] == p[None, ~y])
    numerator = np.einsum('bi,ij,bj->b', pos, kernel, neg, optimize=True)
    denominator = pos.sum(1)*neg.sum(1)
    return np.divide(numerator, denominator, out=np.full(len(draws), np.nan), where=denominator>0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('artifacts/semantic_attribute_data_v1_20261021'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--extended-controls', type=Path)
    p.add_argument('--ego-history', choices=['full','compact'])
    p.add_argument('--encoder-ablations', action='store_true')
    p.add_argument('--extra-features',nargs='*',default=[],help='name=directory with features.npz and source-indexed objects.json')
    p.add_argument('--only-extra',action='store_true')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    data = []
    for split, vf, pf in [
        ('development', 'artifacts/object_features_20261021', 'artifacts/pixel_features_v1_20261021'),
        ('previously_used_holdout', 'artifacts/composition_holdout_features_v1_20261021', 'artifacts/semantic_holdout_pixels_v1_20261021')]:
        with np.load(args.data/f'{split}.npz') as z:
            d = {k: z[k][z['slow_vehicle']] for k in ['object_indices','numeric','attributes','logs','scenes','past_speed']}
        for name, directory in [('vla', vf), ('pixels', pf)]:
            with np.load(Path(directory)/'features.npz') as z:
                d[name] = z['features'][d['object_indices']].reshape(len(d['object_indices']), -1).astype(float)
                if name == 'vla' and args.extended_controls:
                    full = z['features'].reshape(len(z['features']), -1).astype(float)
                    all_objects = json.loads((Path(directory)/'objects.json').read_text())
                    all_scenes = np.array([o['scene'] for o in all_objects])
                    means = {s: full[all_scenes == s].mean(0) for s in set(d['scenes'])}
                    d['vla_scene'] = np.stack([means[s] for s in d['scenes']])
        if args.extended_controls:
            with np.load(args.extended_controls/f'{split}.npz') as z:
                np.testing.assert_array_equal(d['object_indices'], z['object_indices'])
                for name, key in [('clip','features'),('clipcrop','crop'),('clipcontext','context')]:d[name] = z[key].astype(float)
        if args.encoder_ablations:
            for name in ['blind','zeroego','blindzero']:
                directory=Path(f'artifacts/semantic_{name}_features_v1_20261021')
                objects=json.loads((directory/'objects.json').read_text())
                lookup={(o['source_split'],o['source_object_index']):i for i,o in enumerate(objects)}
                indices=[lookup[split,int(i)] for i in d['object_indices']]
                with np.load(directory/'features.npz') as z:d[name]=z['features'][indices].reshape(len(indices),-1).astype(float)
        for entry in args.extra_features:
            name,directory=entry.split('=',1);directory=Path(directory)
            objects=json.loads((directory/'objects.json').read_text())
            lookup={(o['source_split'],o['source_object_index']):i for i,o in enumerate(objects)}
            indices=[lookup[split,int(i)] for i in d['object_indices']]
            with np.load(directory/'features.npz') as z:
                x=z['features'][indices].reshape(len(indices),-1).astype(float)
            mean={s:x[d['scenes']==s].mean(0) for s in set(d['scenes'])}
            d[name]=x;d[name+'mean']=np.stack([mean[s] for s in d['scenes']]);d[name+'center']=x-d[name+'mean']
        d['y'] = d['attributes'] == 'vehicle.stopped'; data.append(d)
        if args.ego_history:
            history = {}; root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
            for scene in set(d['scenes']):
                with np.load(root/scene/f'{scene}_data.npz') as z:
                    xyz=z['ego_history_xyz'].astype(float); rot=z['ego_history_rot'].astype(float).reshape(-1,3,3)
                if args.ego_history=='full':
                    history[scene]=np.r_[xyz.reshape(-1),rot.reshape(-1)]
                else:
                    yaw=np.unwrap(np.arctan2(rot[:,1,0],rot[:,0,0]))
                    velocities=np.stack([(xyz[-1,:2]-xyz[-1-lag,:2])/(lag*.1) for lag in [1,4,8,15]])
                    yaw_rates=np.array([(yaw[-1]-yaw[-1-lag])/(lag*.1) for lag in [1,4,8,15]])
                    history[scene]=np.r_[velocities.reshape(-1),yaw_rates,
                        (xyz[-1,:2]-2*xyz[-5,:2]+xyz[-9,:2])/.4**2]
            d['numeric']=np.c_[d['numeric'],np.stack([history[s] for s in d['scenes']])]
    dev, hold = data; rng = np.random.default_rng(20261021)
    for d in data:d['noise'] = rng.normal(size=(len(d['y']),16))
    feature_names = ['vla','pixels'] + (['vla_scene','clip','clipcrop','clipcontext'] if args.extended_controls else [])
    if args.encoder_ablations:feature_names+=['blind','zeroego','blindzero']
    extra_names=[entry.split('=',1)[0]+suffix for entry in args.extra_features for suffix in ['','mean','center']]
    feature_names=([] if args.only_extra else feature_names)+extra_names
    inputs=['numeric']+extra_names if args.only_extra else ['numeric','vla','pixels','noise']+feature_names[2:]
    methods = ['speed']+[f'{f}_{v}' for f in ['logistic','rbf'] for v in inputs]
    records=[]; hold_predictions={m: [] for m in methods}; cv_predictions={m: np.zeros(len(dev['y'])) for m in methods}
    cv_thresholds={m: np.zeros(len(dev['y'])) for m in methods}
    for fold in range(3):
        split = json.loads(Path(f'artifacts/structured_geometry_log_validation_20261021/fold{fold}_split.json').read_text())
        ti, vi, ei = [np.flatnonzero(np.isin(dev['object_indices'], split[s+'_objects'])) for s in ['train','validation','test']]
        assert not (set(dev['logs'][ti]) & (set(dev['logs'][vi]) | set(dev['logs'][ei]) | set(hold['logs'])))
        assert all(len(set(dev['y'][i])) == 2 for i in [ti, vi, ei])
        base = standardize(dev['numeric'][ti], dev['numeric'][vi], dev['numeric'][ei], hold['numeric'])
        extras = {}
        for name in feature_names:
            mean = dev[name][ti].mean(0); _, _, vt = np.linalg.svd(dev[name][ti]-mean, full_matrices=False)
            extras[name] = standardize((dev[name][ti]-mean)@vt[:16].T, (dev[name][vi]-mean)@vt[:16].T,
                                       (dev[name][ei]-mean)@vt[:16].T, (hold[name]-mean)@vt[:16].T)
        extras['noise'] = standardize(dev['noise'][ti], dev['noise'][vi], dev['noise'][ei], hold['noise'])
        for method in methods:
            if method == 'speed':
                t = threshold(dev['y'][vi], dev['past_speed'][vi]); outputs = [dev['past_speed'][ei], hold['past_speed']]
                selected = {'input': 'past speed', 'validation_auc': auc(dev['y'][vi], dev['past_speed'][vi])}
            else:
                family, features = method.split('_',1)
                values = base if features == 'numeric' else [np.c_[a,b] for a,b in zip(base, extras[features])]
                outputs, t, selected = choose(family, values, dev['y'][ti], dev['y'][vi])
            cv_predictions[method][ei] = outputs[0]; cv_thresholds[method][ei] = t
            hold_predictions[method].append(outputs[1])
            for tag,y,pred,scenes in [('cv',dev['y'][ei],outputs[0],dev['scenes'][ei]),('holdout',hold['y'],outputs[1],hold['scenes'])]:
                records.append({'fold': fold, 'method': method, 'evaluation': tag, 'threshold': t, 'selection': selected,
                                **metrics(y,pred,t), **within_scene_auc(y,pred,scenes)})
        print(f'Completed fold {fold}', flush=True)
    draws = rng.integers(len(set(hold['logs'])), size=(10000,len(set(hold['logs']))))
    # A draw without either class is undefined for every fold, so preserve NaN.
    samples = {m: np.mean([bootstrap_auc(hold['y'], p, hold['logs'], draws) for p in pp], axis=0) for m,pp in hold_predictions.items()}
    comparisons={}
    anchor=extra_names[0] if args.only_extra else 'vla'
    for family in ['logistic', 'rbf']:
        for baseline in [n for n in inputs if n!=anchor]:
            delta=samples[family+'_'+anchor]-samples[family+'_'+baseline]; valid=np.isfinite(delta)
            point=np.mean([auc(hold['y'],p) for p in hold_predictions[family+'_'+anchor]])-np.mean([auc(hold['y'],p) for p in hold_predictions[family+'_'+baseline]])
            comparisons[family+'_'+anchor+'_minus_'+baseline]={'mean_auc_difference':float(point),'conditional_log_bootstrap_95':np.quantile(delta[valid],[.025,.975]).tolist(),'valid_draws':int(valid.sum())}
    summary={'status':'Exploratory semantic decodability only, not a new method or driving gain',
        'development_agents':len(dev['y']),'holdout_agents':len(hold['y']),'holdout_logs':len(set(hold['logs'])),
        'holdout_mean_auc':{m:float(np.mean([r['auc'] for r in records if r['method']==m and r['evaluation']=='holdout'])) for m in methods},
        'holdout_mean_balanced_error':{m:float(np.mean([r['balanced_error'] for r in records if r['method']==m and r['evaluation']=='holdout'])) for m in methods},
        'cv_fold_mean_auc':{m:float(np.mean([r['auc'] for r in records if r['method']==m and r['evaluation']=='cv'])) for m in methods},
        'holdout_comparisons':comparisons,
        'limitations':['Previously examined data and evaluation logs','Small48-agent12-log evaluation','Annotation proxy, not verified intention','No evidence of action use or future-departure prediction','No multiple-selection correction or retraining bootstrap']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(args.out/'predictions.npz',holdout_labels=hold['y'],holdout_logs=hold['logs'],development_labels=dev['y'],
                        **{'holdout_'+m:np.array(pp) for m,pp in hold_predictions.items()},**{'cv_'+m:pp for m,pp in cv_predictions.items()})
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_ESTADO_SEMANTICO_V1.md').read_bytes()).hexdigest(),
        'seed':20261021,'selection':'Validation AUC, then threshold validation BER; no evaluation selection','families':methods,
        'ego_history_control':args.ego_history,'numeric_dimensions':int(dev['numeric'].shape[1]),
        'encoder_ablations':args.encoder_ablations,'extra_features':args.extra_features,'only_extra':args.only_extra},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
