"""Reviewer 2, round 4: power of the matched-temperature test against contractions with the wrong mechanism.

For each model and k in (4,10) we build spread-matched controls (c or p bisected so that the clip-mean
spread equals E_k's exactly), compute residuals vs the explicit-temperature curve exactly as
paper_numbers.matched does, and bootstrap clips jointly across models (1000 resamples, seed 20261026).
Also: pooled residuals, paired Euler-minus-shrink residuals, and the A1.5 holdout clips 160-199.
"""
import json, sys
from pathlib import Path
import numpy as np
from cavla import paper_numbers as pn
from cavla.trajectory_decision_gate import decisions

A = Path('/mnt/datasets/CAVLA/artifacts'); D = A / 'physicalai_test_contexts_v1_20261022'
REC = {r['name']: r for r in json.loads((D / 'records.json').read_text()) if 'error' not in r}
TAUS = pn.TAUS; CURVE = pn.CURVE; REF = pn.REFERENCE
METRICS = ['ade1', 'minade6', 'flip', 'entropy']


def spread(ts): return pn.spread(ts)


def build_controls(P, names, seeds, k):
    """Return dict cfg -> function(c) that writes plans for all names/seeds and the cfg name."""
    T = np.mean([spread([P[f'{n}_s{s}_euler{k}_t1.0'] for s in seeds]) for n in names])
    rng = np.random.default_rng(7)
    U = {(n, s): rng.random() for n in names for s in seeds}
    meanA = {n: np.mean([P[f'{n}_s{s}_{REF}'] for s in seeds], 0) for n in names}
    t = 0.1 * np.arange(1, 65)
    cv = {n: np.stack([REC[n]['speed_mps'] * t, 0 * t, 0 * t], 1) for n in names}

    def make(kind, c):
        out = {}
        for n in names:
            for s in seeds:
                a = P[f'{n}_s{s}_{REF}']; e1 = P[f'{n}_s{s}_euler1_t1.0']
                if kind == 'shrinkE1': x = e1 + c * (a - e1)
                elif kind == 'shrinkMeanA': x = meanA[n] + c * (a - meanA[n])
                elif kind == 'shrinkCV': x = cv[n] + c * (a - cv[n])
                elif kind == 'mixE1': x = e1 if U[(n, s)] < c else a      # c = collapse probability
                elif kind == 'shrinkX':
                    x = a.copy(); x[:, 0] = e1[:, 0] + c * (a[:, 0] - e1[:, 0])
                elif kind == 'shrinkY':
                    x = a.copy(); x[:, 1] = e1[:, 1] + c * (a[:, 1] - e1[:, 1])
                out[f'{n}_s{s}_{kind}{k}_t1.0'] = x
        return out

    def msp(d): return np.mean([spread([d[f'{n}_s{s}_' + key] for s in seeds]) for n in names])
    res = {}
    for kind in ['shrinkE1', 'shrinkMeanA', 'shrinkCV', 'mixE1', 'shrinkX', 'shrinkY']:
        key = f'{kind}{k}_t1.0'
        lo, hi = 0., 1.
        f = lambda c: msp(make(kind, c))
        decreasing = kind != 'mixE1'  # spread increases with c for shrinks, decreases with p for mix
        flo, fhi = f(lo), f(hi)
        if not (min(flo, fhi) <= T <= max(flo, fhi)):
            res[key] = (None, flo, fhi, T); continue
        for _ in range(30):
            mid = (lo + hi) / 2; fm = f(mid)
            if (fm < T) == decreasing: lo = mid
            else: hi = mid
        c = (lo + hi) / 2; P.update(make(kind, c)); res[key] = (c, f(c), None, T)
    # paper's shrink (pooled c from ratio of means)
    return res


def load_model(probe, names):
    seeds = json.loads((A / probe / 'manifest.json').read_text())['seeds']
    with np.load(A / probe / 'plans.npz') as z: P = {k: z[k] for k in z.files}
    pn.add_shrinkage(P, names, seeds)
    info = {}
    for k in (4, 10): info[k] = build_controls(P, names, seeds, k)
    cfgs = sorted({k.split('_', 3)[-1] for k in P if k.startswith(names[0] + '_s')})
    tab = {}
    for n in names:
        human = np.load(D / f'{n}.npz')['ego_future_xyz']; sp = REC[n]['speed_mps']
        ref_dec = [decisions(P[f'{n}_s{s}_{REF}'], sp) for s in seeds]
        for c in cfgs:
            if c.startswith('euler10_t') and c != 'euler10_t1.0': continue
            if c.startswith(('heun3_t', 'heun5_t')) and not c.endswith('t1.0'): continue
            tab.setdefault(c, {})[n] = pn.clip_metrics(P, n, c, seeds, sp, human, ref_dec)
    arr = {c: {m: np.array([tab[c][n][m] for n in names]) for m in METRICS + ['spread']} for c in tab}
    e1sp = np.mean([spread([P[f'{n}_s{s}_euler1_t1.0'] for s in seeds]) for n in names])
    return arr, info, e1sp, tab


def matched(arr, idx, sampler):
    sp = np.array([arr[c]['spread'][idx].mean() for c in CURVE]); o = np.argsort(sp); s = arr[sampler]['spread'][idx].mean()
    out = {'tau': float(np.interp(s, sp[o], np.array(TAUS)[o]))}
    for m in METRICS:
        cur = np.array([arr[c][m][idx].mean() for c in CURVE]); at = float(np.interp(s, sp[o], cur[o]))
        out[m] = arr[sampler][m][idx].mean() - at
    return out


def run(names, models, samplers, B=1000, label=''):
    data = {mk: load_model(pr, names) for mk, pr in models}
    rng = np.random.default_rng(20261026); N = len(names)
    idxs = [np.arange(N)] + [rng.choice(N, N) for _ in range(B)]
    print(f'\n######## {label}: {N} clips, {B} joint resamples')
    for mk in data:
        arr, info, e1sp, _ = data[mk]
        print(f'== {mk}  E1 spread {e1sp:.3f} m  A1 spread {arr[REF]["spread"].mean():.3f} m  E4 {arr["euler4_t1.0"]["spread"].mean():.3f}  E10 {arr["euler10_t1.0"]["spread"].mean():.3f}  paper-shrink4 {arr["shrink4_t1.0"]["spread"].mean():.3f}  shrink10 {arr["shrink10_t1.0"]["spread"].mean():.3f}')
        for k in info:
            for key, v in info[k].items(): print(f'   control {key}: c={v[0]}  (spread {v[1]:.3f}, target {v[3]:.3f})' if v[0] is not None else f'   control {key}: cannot reach target {v[3]:.3f} (range {v[1]:.3f}-{v[2]:.3f})')
    results = {}
    for samp in samplers:
        per = {mk: [matched(data[mk][0], i, samp) for i in idxs] for mk in data if samp in data[mk][0]}
        if not per: continue
        results[samp] = per
        for mk, runs in per.items():
            p = runs[0]; b = runs[1:]
            q = lambda m: np.quantile([r[m] for r in b], [.025, .975])
            q90 = np.quantile([r['flip'] for r in b], [.05, .95])
            print(f'{samp:16s} {mk:11s} tau {p["tau"]:.3f} | flip res {100*p["flip"]:+5.2f} [{100*q("flip")[0]:+5.2f},{100*q("flip")[1]:+5.2f}] (90%: [{100*q90[0]:+.2f},{100*q90[1]:+.2f}]) | ADE1 res {p["ade1"]:+.3f} [{q("ade1")[0]:+.3f},{q("ade1")[1]:+.3f}] | minADE6 res {p["minade6"]:+.3f} [{q("minade6")[0]:+.3f},{q("minade6")[1]:+.3f}] | ent {p["entropy"]:+.3f}')
        if len(per) == 3:
            pooled = lambda m, j: np.mean([per[mk][j][m] for mk in per])
            for m in ['flip', 'ade1', 'minade6']:
                vals = [pooled(m, j) for j in range(len(idxs))]; sc = 100 if m == 'flip' else 1
                lo, hi = np.quantile(vals[1:], [.025, .975])
                print(f'{samp:16s} POOLED      {m:7s} res {sc*vals[0]:+.3f} [{sc*lo:+.3f},{sc*hi:+.3f}]')
    return data, results, idxs


def paired(results, a, b, idxs):
    """Pooled residual difference a - b over the three models (same resamples)."""
    for m in ['flip', 'ade1', 'minade6']:
        sc = 100 if m == 'flip' else 1
        for mk in results[a]:
            v = [results[a][mk][j][m] - results[b][mk][j][m] for j in range(len(idxs))]
            lo, hi = np.quantile(v[1:], [.025, .975]); print(f'  {a}-{b} {mk:11s} {m:7s} {sc*v[0]:+.3f} [{sc*lo:+.3f},{sc*hi:+.3f}]')
        v = [np.mean([results[a][mk][j][m] - results[b][mk][j][m] for mk in results[a]]) for j in range(len(idxs))]
        lo, hi = np.quantile(v[1:], [.025, .975]); print(f'  {a}-{b} POOLED      {m:7s} {sc*v[0]:+.3f} [{sc*lo:+.3f},{sc*hi:+.3f}]')


if __name__ == '__main__':
    MODELS = [('alpamayo15', 'temperature_probe_v1_20261022'), ('alpamayo_r1', 'temperature_probe_r1_v1_20261022'), ('alpamayo2', 'temperature_probe_a2_v1_20261022')]
    which = sys.argv[1]
    if which == 'main':
        names = [f'context_{i:03d}' for i in range(100, 160)]
        S = []
        for k in (4, 10): S += [f'euler{k}_t1.0', f'shrink{k}_t1.0'] + [f'{c}{k}_t1.0' for c in ['shrinkE1', 'shrinkMeanA', 'shrinkCV', 'mixE1', 'shrinkX', 'shrinkY']]
        S += ['heun3_t1.0', 'heun5_t1.0']
        data, res, idxs = run(names, MODELS, S, label='paper sample 100-159')
        print('\nPaired residual differences')
        for k in (4, 10): paired(res, f'euler{k}_t1.0', f'shrink{k}_t1.0', idxs)
    else:
        lo, hi = map(int, which.split('-'))
        names = [f'context_{i:03d}' for i in range(lo, hi)]
        S = ['euler4_t1.0', 'euler10_t1.0', 'shrink4_t1.0', 'shrink10_t1.0', 'heun3_t1.0', 'heun5_t1.0', 'euler6_t1.0']
        run(names, MODELS[:1], S, label=f'A1.5 clips {lo}-{hi-1}')
