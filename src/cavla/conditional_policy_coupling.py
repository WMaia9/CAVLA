"""Analytic controlled coupling gate. Classical conditional Gaussian transport.

Preserves specified one-time state/action marginals, not the full history law,
first-exit probabilities, or an arbitrary partially observed policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = ['iid','ar_noise','marginal_whitening','action_ema','conditional_09','conditional_1','conditional_observed_previous']
A = np.array([[1.,.1],[0.,.95]])
B = np.array([.005,.1])
K = np.array([-1.,-1.5])
SIGMA = .3


def controller(method, covariance):
    """State is [position,velocity,previous_action,previous_noise]."""
    row = np.zeros(4); row[:2] = K; fresh = SIGMA
    if method == 'ar_noise':
        row[3] = SIGMA*.9; fresh = SIGMA*np.sqrt(1-.9**2)
    elif method == 'action_ema':
        row[:2] *= .1; row[2] = .9; fresh *= .1
    elif method == 'marginal_whitening':
        variance = covariance[2,2]
        if variance > 1e-12: row[2] = SIGMA/np.sqrt(variance); fresh = 0.
    elif method.startswith('conditional_'):
        rho = .9 if method == 'conditional_09' else 1.
        dims = 3 if method == 'conditional_observed_previous' else 2
        regression = np.linalg.pinv(covariance[:dims,:dims],rcond=1e-12)@covariance[:dims,2]
        variance = covariance[2,2]-covariance[2,:dims]@regression
        if variance > 1e-12:
            scale = SIGMA*rho/np.sqrt(variance)
            row[:dims] -= scale*regression; row[2] += scale
            fresh = SIGMA*np.sqrt(1-rho**2)
    elif method != 'iid': raise ValueError(method)
    return row, fresh


def step_matrices(row,fresh,process_std):
    transform = np.zeros((4,4)); transform[:2,:2] = A
    transform[:2] += B[:,None]*row[None,:]; transform[2] = row
    transform[3] = (row-np.r_[K,0.,0.])/SIGMA
    noise = np.zeros((4,3)); noise[:2,0] = B*fresh; noise[2,0] = fresh
    noise[3,0] = fresh/SIGMA; noise[:2,1:] = process_std*np.eye(2)
    return transform,noise


def analytic(method,process_std,steps=100):
    covariance = np.diag([.01,.01,0.,0.]); records = []; transforms = []
    for t in range(steps):
        row,fresh = controller(method,covariance)
        pxx = covariance[:2,:2]
        cross = covariance[:2]@row
        action_var = row@covariance@row+fresh**2
        slope = np.linalg.solve(pxx,cross)
        conditional_var = max(0.,action_var-cross@slope)
        difference = row.copy(); difference[2] -= 1
        records.append({'time':t,'state_covariance':pxx.tolist(),'action_variance':float(action_var),
            'conditional_action_mean_slope':slope.tolist(),'conditional_action_variance':float(conditional_var),
            'action_difference_variance':float(difference@covariance@difference+fresh**2)})
        transform,noise = step_matrices(row,fresh,process_std)
        transforms.append((transform,noise))
        covariance = transform@covariance@transform.T+noise@noise.T
        covariance = (covariance+covariance.T)/2
        if np.linalg.eigvalsh(covariance).min() < -1e-9: raise ArithmeticError('Invalid covariance')
    return records,transforms


def simulate(transforms,seed,draws=50000,threshold=.35):
    rng = np.random.default_rng(seed); state = np.zeros((draws,4)); state[:,:2] = rng.normal(0,.1,(draws,2))
    exited = np.abs(state[:,0]) > threshold
    for transform,noise in transforms:
        state = state@transform.T+rng.normal(size=(draws,3))@noise.T
        exited |= np.abs(state[:,0]) > threshold
    return float(exited.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--draws',type=int,default=50000); args = p.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    source = Path(__file__).read_bytes(); (args.out/'source_snapshot.py').write_bytes(source)
    protocol = Path('docs/PROTOCOLO_ACOPLAMENTO_POLITICA_V1.md').read_bytes(); (args.out/'protocol_snapshot.md').write_bytes(protocol)
    results = []
    for process_std in [0.,.005,.02,.1]:
        reference,_ = analytic('iid',process_std)
        for method in METHODS:
            records,transforms = analytic(method,process_std)
            state_error = max(np.max(np.abs(np.array(a['state_covariance'])-np.array(b['state_covariance']))) for a,b in zip(records,reference))
            action_error = max(abs(a['action_variance']-b['action_variance']) for a,b in zip(records,reference))
            mean_error = max(np.max(np.abs(np.array(a['conditional_action_mean_slope'])-K)) for a in records)
            variance_error = max(abs(a['conditional_action_variance']-SIGMA**2) for a in records)
            if method.startswith('conditional_'):
                assert max(state_error,action_error,mean_error,variance_error) < 1e-8
            risk = [simulate(transforms,seed,args.draws) for seed in [17,29,43,71,101]]
            result = {'process_std':process_std,'method':method,'maximum_state_covariance_difference_from_iid':float(state_error),
                'maximum_action_variance_difference_from_iid':float(action_error),'maximum_conditional_mean_slope_error':float(mean_error),
                'maximum_conditional_variance_error':float(variance_error),
                'mean_action_difference_variance':float(np.mean([a['action_difference_variance'] for a in records])),
                'reference_action_difference_variance':float(np.mean([a['action_difference_variance'] for a in reference])),
                'mean_position_variance':float(np.mean([a['state_covariance'][0][0] for a in records])),
                'first_exit_probability_by_seed':risk,'first_exit_probability_mean':float(np.mean(risk)),
                'draws_per_seed':args.draws,'records':records}
            results.append(result); print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    (args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'source_sha256':hashlib.sha256(source).hexdigest(),
        'protocol_sha256':hashlib.sha256(protocol).hexdigest(),'A':A.tolist(),'B':B.tolist(),'K':K.tolist(),'sigma':SIGMA,'results':results},indent=2))


if __name__=='__main__':main()
