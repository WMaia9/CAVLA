"""Exploratory information probe beyond past-only constant velocity; numpy only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def standardize(train, *others):
    mean = train.mean(0); scale = train.std(0).clip(min=1e-6)
    return [(x - mean) / scale for x in (train, *others)]


def log_mean(values, logs):
    return float(np.mean([values[logs == log].mean() for log in sorted(set(logs))]))


def distance(x, y):
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None] - 2*x@y.T, 0) / x.shape[1]


def select_predict(family, x, y, validation, evaluation, val_logs, val_target):
    # Every family can choose the unmodified kinematic predictor.
    best = log_mean(np.linalg.norm(val_target.reshape(-1, 64, 2), axis=-1).mean(1), val_logs)
    answer = [np.zeros((len(a), y.shape[1])) for a in evaluation]
    selected = {'correction': 'zero', 'validation_log_ade': best}
    candidates = []
    alphas = [.01, .1, 1., 10., 100., 1000.] if family == 'linear' else [.001, .01, .1, 1., 10.]
    scales = [None] if family == 'linear' else [.25, 1., 4.]
    for bandwidth in scales:
        if family == 'linear':
            tx = np.c_[np.ones(len(x)), x]; vx = np.c_[np.ones(len(validation)), validation]
            ex = [np.c_[np.ones(len(a)), a] for a in evaluation]
            matrix = tx.T @ tx; rhs = tx.T @ y
        else:
            matrix = np.exp(-distance(x, x) / bandwidth)
            vx = np.exp(-distance(validation, x) / bandwidth)
            ex = [np.exp(-distance(a, x) / bandwidth) for a in evaluation]
            rhs = y
        eig, basis = np.linalg.eigh(matrix)
        for alpha in alphas:
            coefficient = basis @ ((basis.T @ rhs) / (np.maximum(eig, 0)[:, None] + alpha))
            residual = (vx @ coefficient - val_target).reshape(-1, 64, 2)
            score = log_mean(np.linalg.norm(residual, axis=-1).mean(1), val_logs)
            candidates.append({'alpha': alpha, 'bandwidth': bandwidth, 'validation_log_ade': score})
            if score < best:
                best = score; answer = [a @ coefficient for a in ex]
                selected = {'correction': family, 'alpha': alpha, 'bandwidth': bandwidth, 'validation_log_ade': score}
    selected['fitted_candidates'] = candidates
    return answer, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    loaded = []
    for name, feature_path in [('development', 'artifacts/object_features_20261021/features.npz'),
                               ('previously_used_holdout', 'artifacts/composition_holdout_features_v1_20261021/features.npz')]:
        with np.load(args.data / f'{name}_labels_predictions.npz') as z:
            d = {k: z[k] for k in ['eligible_past', 'object_indices', 'numeric', 'future_centers', 'logs', 'scenes']}
        use = d['eligible_past']; d = {k: v[use] for k, v in d.items() if k != 'eligible_past'}
        with np.load(feature_path) as z:
            d['visual'] = z['features'][d['object_indices']].reshape(len(d['object_indices']), -1).astype(float)
        time = np.arange(1, 65) * .1
        # numeric: 4 bbox + 6 classes + center_xy + velocity_xy + cos/sin + size.
        d['cv'] = d['numeric'][:, 10:12, None].transpose(0, 2, 1) + time[None, :, None] * d['numeric'][:, None, 12:14]
        d['target'] = (d['future_centers'] - d['cv']).reshape(len(d['numeric']), -1)
        loaded.append(d)
    dev, hold = loaded; rng = np.random.default_rng(20261021)
    noise_dev = rng.normal(size=(len(dev['numeric']), 16)); noise_hold = rng.normal(size=(len(hold['numeric']), 16))
    rows = []; errors = {name: [] for name in ['constant_velocity', 'linear_numeric', 'linear_vla', 'linear_noise', 'rbf_numeric', 'rbf_vla', 'rbf_noise']}
    cv_errors = {name: np.zeros((len(dev['numeric']), 64)) for name in errors}
    for fold in range(3):
        split = json.loads(Path(f'artifacts/structured_geometry_log_validation_20261021/fold{fold}_split.json').read_text())
        ti = np.flatnonzero(np.isin(dev['object_indices'], split['train_objects']))
        vi = np.flatnonzero(np.isin(dev['object_indices'], split['validation_objects']))
        ei = np.flatnonzero(np.isin(dev['object_indices'], split['test_objects']))
        assert not (set(dev['logs'][ti]) & (set(dev['logs'][vi]) | set(dev['logs'][ei]) | set(hold['logs'])))
        base = standardize(dev['numeric'][ti], dev['numeric'][vi], dev['numeric'][ei], hold['numeric'])
        mean = dev['visual'][ti].mean(0)
        _, _, vt = np.linalg.svd(dev['visual'][ti] - mean, full_matrices=False)
        visual = standardize((dev['visual'][ti]-mean) @ vt[:16].T,
                             (dev['visual'][vi]-mean) @ vt[:16].T,
                             (dev['visual'][ei]-mean) @ vt[:16].T,
                             (hold['visual']-mean) @ vt[:16].T)
        noise = standardize(noise_dev[ti], noise_dev[vi], noise_dev[ei], noise_hold)
        for name in errors:
            if name == 'constant_velocity':
                pred = [np.zeros_like(dev['target'][ei]), np.zeros_like(hold['target'])]
                selected = {'correction': 'zero'}
            else:
                family, features = name.split('_')
                values = base if features == 'numeric' else [np.c_[a, b] for a, b in zip(base, visual if features == 'vla' else noise)]
                pred, selected = select_predict(family, values[0], dev['target'][ti], values[1], values[2:], dev['logs'][vi], dev['target'][vi])
            for tag, target, prediction, logs in [('cv', dev['target'][ei], pred[0], dev['logs'][ei]), ('holdout', hold['target'], pred[1], hold['logs'])]:
                error = np.linalg.norm((target - prediction).reshape(-1, 64, 2), axis=-1)
                rows.append({'fold': fold, 'method': name, 'evaluation': tag, 'agents': len(error), 'logs': len(set(logs)),
                             'log_mean_ade_m': log_mean(error.mean(1), logs), 'agent_mean_ade_m': float(error.mean()),
                             'log_mean_fde_m': log_mean(error[:, -1], logs), 'selection': selected})
                if tag == 'holdout': errors[name].append(error)
                else: cv_errors[name][ei] = error
        print(f'Completed fold {fold}', flush=True)
    mean_errors = {m: np.mean(e, axis=0) for m, e in errors.items()}
    hold_logs = sorted(set(hold['logs'])); draws = rng.integers(len(hold_logs), size=(10000, len(hold_logs)))
    comparisons = {}
    for family in ['linear', 'rbf']:
        for baseline in ['numeric', 'noise']:
            delta = mean_errors[family+'_vla'].mean(1) - mean_errors[family+'_'+baseline].mean(1)
            values = np.array([delta[hold['logs'] == log].mean() for log in hold_logs])
            comparisons[family+'_vla_minus_'+baseline] = {'log_mean_ade_difference_m': float(values.mean()), 'conditional_log_bootstrap_95': np.quantile(values[draws].mean(1), [.025, .975]).tolist()}
    summary = {'status': 'Exploratory information probe, no novel method or independent confirmation',
               'development_agents': len(dev['numeric']), 'holdout_agents': len(hold['numeric']),
               'holdout_logs': len(hold_logs),
               'holdout_log_mean_ade_m': {m: log_mean(e.mean(1), hold['logs']) for m, e in mean_errors.items()},
               'cv_log_mean_ade_m': {m: log_mean(e.mean(1), dev['logs']) for m, e in cv_errors.items()},
               'holdout_comparisons': comparisons,
               'limitations': ['All evaluation sets previously examined', 'Privileged past 3D inputs and supervised 3D futures', 'Fixed-model bootstrap only', 'No action selection, edited pairs, reactive futures, or safety validation', 'PCA16 only; negative probe does not establish absence of all visual information']}
    np.savez_compressed(args.out/'errors.npz', holdout_logs=hold['logs'], development_logs=dev['logs'],
                        **{'holdout_'+m: e for m, e in mean_errors.items()}, **{'cv_'+m: e for m, e in cv_errors.items()})
    (args.out/'records.json').write_text(json.dumps(rows, indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'manifest.json').write_text(json.dumps({'data': str(args.data), 'selection': 'Inner-validation log-mean ADE, never evaluation',
        'protocol_sha256': hashlib.sha256(Path('docs/PROTOCOLO_INOVACAO_TEMPORAL_V1.md').read_bytes()).hexdigest(),
        'deterministic': True, 'seed': 20261021, 'families': 'Linear ridge, RBF kernel ridge, numeric/VLA16/Gaussian16; zero correction eligible in every family'}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
