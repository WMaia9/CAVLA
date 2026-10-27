"""Known-graph restricted rate check; adaptive classical feedback is a control."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from cavla.maxplus_capacity_probe import band_edges
from cavla.maxplus_intervention_probe import closure
from cavla.maxplus_quantization_probe import encode, path_response_error


def adaptive_encode(a, order, epsilon, method):
    n = len(a)
    if not 0 <= order <= n-2 or not 0 < epsilon < .2:
        raise ValueError('Outside the proved domain')
    weights = a[np.tril_indices(n, -1)]
    if not np.all((weights >= 1) & (weights <= 1.4)):
        raise ValueError('Weights outside [1,1.4]')
    if method not in ['potential', 'chain']:
        raise ValueError(method)
    delta0 = epsilon/2 if method == 'potential' else epsilon/(2*(order+1))
    s = np.r_[0., np.cumsum(np.diag(a, -1))]
    shat = delta0*np.rint(s/delta0); e = shat-s
    coded = np.full_like(a, -np.inf); payload = 0
    for v, u in band_edges(n, order):
        r = v-u-1
        delta = epsilon*r/order if r else delta0
        if r == 0:
            value = shat[v]-shat[u]
        else:
            value = a[v, u] + (e[v]-e[u] if method == 'potential' else 0)
        coded[v, u] = delta*np.rint(value/delta)
        # Common broad range for both constructions; no hidden per-graph header.
        lo = int(np.floor((1-epsilon)/delta)); hi = int(np.ceil((1.4+epsilon)/delta))
        code = int(np.rint(coded[v, u]/delta)); assert lo <= code <= hi
        payload += int(np.ceil(np.log2(hi-lo+1)))
    return coded, payload


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes(); (args.out/'source_snapshot.py').write_bytes(source)
    protocol = Path('docs/PROTOCOLO_CODIFICACAO_REMOCOES_V1.md').read_bytes()
    (args.out/'protocol_snapshot.md').write_bytes(protocol)
    rng = np.random.default_rng(20261022); epsilon = .02; records = []
    for n in [8,32,128,256]:
        for k in [0,1,2,4,8,16,32,64]:
            if k > n-2: continue
            removed_sets = [r for size in range(k+1) for r in itertools.combinations(range(n),size)] if n == 8 else sorted(
                {()} | {tuple(sorted(map(int,rng.choice(n,size,replace=False)))) for size in range(1,k+1) for _ in range(64)})
            errors = {name: [] for name in ['uniform_potential','adaptive_potential','adaptive_chain']}; payloads = {}
            for seed in range(5):
                a = np.full((n,n),-np.inf); ix = np.tril_indices(n,-1)
                a[ix] = rng.uniform(1,1.4,len(ix[0])); delta = 2*epsilon/(k+2)
                uniform = encode(a,k,delta,'potential_feedback')
                lo = int(np.floor((1-1.5*delta)/delta)); hi = int(np.ceil((1.4+1.5*delta)/delta))
                payloads['uniform_potential'] = len(band_edges(n,k))*int(np.ceil(np.log2(hi-lo+1)))
                adaptive, payloads['adaptive_potential'] = adaptive_encode(a,k,epsilon,'potential')
                chain, payloads['adaptive_chain'] = adaptive_encode(a,k,epsilon,'chain')
                for name,b in [('uniform_potential',uniform),('adaptive_potential',adaptive),('adaptive_chain',chain)]:
                    values = [path_response_error(a,b,r) for r in removed_sets]
                    errors[name].append(max(values)); assert max(values) <= epsilon+1e-9
                    if n == 8:
                        for r, value in zip(removed_sets,values):
                            c = closure(a,r); d = closure(b,r); valid = np.isfinite(c)
                            np.testing.assert_array_equal(np.isfinite(d),valid)
                            np.testing.assert_allclose(value,np.max(np.abs(c[valid]-d[valid])),atol=1e-11,rtol=0)
            m = len(band_edges(n,k))
            records.append({'nodes':n,'order':k,'coordinates':m,'epsilon':epsilon,'graphs':5,
                'queries_per_graph':len(removed_sets),'exhaustive_deletions':n==8,
                'covering_lower_bound_bits':m*np.log2(.4/(2*epsilon)),
                'methods':{name:{'payload_bits':payloads[name],'maximum_tested_error':max(vals)} for name,vals in errors.items()}})
    result = {'scope':'Restricted known-graph response coding, no novelty or VLA applicability claim',
        'source_sha256':hashlib.sha256(source).hexdigest(),'protocol_sha256':hashlib.sha256(protocol).hexdigest(),
        'records':records,'limitations':['All methods know latent graph; header n,k,epsilon and fixed domain is shared',
            'Large graph queries are sampled; theorem and empirical maxima must be distinguished',
            'Adaptive chain feedback also has the same asymptotic rate; no exclusive potential-method rate advantage']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps({'configurations':len(records),'passed':True}))


if __name__ == '__main__': main()
