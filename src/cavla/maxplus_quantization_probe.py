"""Restricted deletion-response coding check, including classical feedback controls.

All graphs and latent weights are provided to the encoder. No learned VLA is
tested. Potentials and error feedback are classical; this is not a novelty claim.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from cavla.maxplus_capacity_probe import band_edges
from cavla.maxplus_intervention_probe import closure


def encode(a, order, delta, method):
    n = len(a)
    if not 0 <= order <= n - 2 or not 0 < delta < .6 / 4.5:
        raise ValueError('Outside the proved order or margin condition')
    weights = a[np.tril_indices(n, -1)]
    if not np.all((weights >= 1.) & (weights <= 1.4)):
        raise ValueError('Weights must be in [1,1.4]')
    s = np.r_[0., np.cumsum(np.diag(a, -1))]
    shat = delta * np.rint(s / delta)
    error = shat - s
    out = np.full_like(a, -np.inf)
    for v, u in band_edges(n, order):
        if method == 'independent':
            value = a[v, u]
        elif method == 'chain_feedback':
            value = shat[v] - shat[u] if v == u + 1 else a[v, u]
        elif method == 'potential_feedback':
            value = a[v, u] + error[v] - error[u]
        else:
            raise ValueError(method)
        out[v, u] = delta * np.rint(value / delta)
    return out


def path_response_error(a, coded, removed):
    """All-pair maximum error, assuming both critical paths visit all survivors.

    The domain margin checked by encode guarantees this property. Tests compare
    this O(n) computation against the full closure, rather than assuming it.
    """
    removed = set(removed)
    alive = np.array([i for i in range(len(a)) if i not in removed])
    edge_error = coded[alive[1:], alive[:-1]] - a[alive[1:], alive[:-1]]
    cumulative = np.r_[0., np.cumsum(edge_error)]
    return float(np.ptp(cumulative))


def queries(n, order, rng, draws):
    if n <= 10:
        return [r for size in range(order + 1) for r in itertools.combinations(range(n), size)], True
    result = {()}
    for size in range(1, order + 1):
        # Include contiguous and evenly distributed failures, plus random sets.
        for start in [0, n // 2, n - size]:
            result.add(tuple(range(start, min(n, start + size))))
        result.add(tuple(np.linspace(1, n - 2, size, dtype=int)))
        for _ in range(draws):
            result.add(tuple(sorted(map(int, rng.choice(n, size, replace=False)))))
    return sorted(result), False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.out / 'source_snapshot.py').write_bytes(source)
    rng = np.random.default_rng(20261022)
    delta = .01
    methods = ['independent', 'chain_feedback', 'potential_feedback']
    records = []
    for n in [8, 16, 64, 256]:
        for order in [0, 1, 2, 4]:
            query_list, exhaustive = queries(n, order, rng, 128)
            for population in ['iid_uniform', 'constructed_aligned_rounding']:
                errors = {m: [] for m in methods}
                count = 20 if population == 'iid_uniform' else 1
                for _ in range(count):
                    a = np.full((n, n), -np.inf)
                    ix = np.tril_indices(n, -1)
                    a[ix] = rng.uniform(1., 1.4, len(ix[0])) if population == 'iid_uniform' else 1.2049
                    for method in methods:
                        coded = encode(a, order, delta, method)
                        values = [path_response_error(a, coded, r) for r in query_list]
                        maximum = max(values)
                        if method == 'potential_feedback':
                            assert maximum <= delta * (1 + order / 2) + 1e-10
                        if method == 'chain_feedback':
                            assert maximum <= delta * (1 + 1.5 * order) + 1e-10
                        if exhaustive:
                            for removed, value in zip(query_list, values):
                                c = closure(a, removed); d = closure(coded, removed)
                                finite = np.isfinite(c)
                                np.testing.assert_array_equal(np.isfinite(d), finite)
                                np.testing.assert_allclose(value, np.max(np.abs(c[finite] - d[finite])), atol=1e-11, rtol=0)
                        lo = int(np.floor((1 - 1.5 * delta) / delta))
                        hi = int(np.ceil((1.4 + 1.5 * delta) / delta))
                        codes = np.rint(coded[np.isfinite(coded)] / delta)
                        assert codes.min() >= lo and codes.max() <= hi
                        errors[method].append(maximum)
                m = len(band_edges(n, order)); bits = int(np.ceil(np.log2(hi - lo + 1)))
                records.append({'nodes': n, 'order': order, 'population': population,
                    'graphs': count, 'queries_per_graph': len(query_list), 'exhaustive_deletions': exhaustive,
                    'delta': delta, 'edge_coordinates': m, 'bits_per_coordinate': bits,
                    'payload_bits_each_method': bits * m,
                    'potential_proved_bound': delta * (1 + order / 2),
                    'chain_proved_bound': delta * (1 + 1.5 * order),
                    'methods': {name: {'mean_graph_maximum': float(np.mean(vals)),
                        'maximum_tested_error': float(max(vals))} for name, vals in errors.items()}})
    result = {'scope': 'Known latent restricted graphs, classical quantizers; not VLA compression or a novelty claim',
        'source_sha256': hashlib.sha256(source).hexdigest(), 'seed': 20261022,
        'limitations': ['Common n,k,delta,interval and edge order are shared side information; payload excludes that common header',
            'Large-n maximum is over sampled queries, not an exhaustive maximum',
            'Aligned-rounding construction is a deliberate worst-case example, not a data distribution',
            'Classical consistent graph rounding and feedback coding must be compared before novelty claims'],
        'records': records}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'configurations': len(records), 'all_assertions_passed': True}))


if __name__ == '__main__':
    main()
