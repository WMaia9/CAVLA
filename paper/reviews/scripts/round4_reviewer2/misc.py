import json
from pathlib import Path
from collections import Counter
import numpy as np
from cavla import paper_numbers as pn
from cavla.trajectory_decision_gate import decisions
A = Path('/mnt/datasets/CAVLA/artifacts'); D = A / 'physicalai_test_contexts_v1_20261022'
REC = {r['name']: r for r in json.loads((D / 'records.json').read_text()) if 'error' not in r}
MODELS = [('alpamayo15', 'temperature_probe_v1_20261022'), ('alpamayo_r1', 'temperature_probe_r1_v1_20261022'), ('alpamayo2', 'temperature_probe_a2_v1_20261022')]
S = json.loads((A / 'paper_numbers_v2_20261026/summary.json').read_text())
G = json.loads((A / 'gaussian_contraction_v1_20261026/summary.json').read_text())
d = lambda a, b: np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean()
key = lambda x: (x['longitudinal'], x['lateral'])
def ed(X, Y): return 2*np.mean([d(x, y) for x in X for y in Y]) - np.mean([d(a, b) for a in X for b in X]) - np.mean([d(a, b) for a in Y for b in Y])
def tv(X, Y, sp):
    cx = Counter(key(decisions(x, sp)) for x in X); cy = Counter(key(decisions(y, sp)) for y in Y)
    return .5*sum(abs(cx[k]-cy[k]) for k in set(cx) | set(cy))/len(X)
print('Gaussian predicted medians', {k: G['steps'][k]['predicted_median'] for k in G['steps']})
for rng_ in [(100, 160), (160, 200)]:
    names = [f'context_{i:03d}' for i in range(*rng_)]
    for mk, pr in MODELS:
        seeds = json.loads((A / pr / 'manifest.json').read_text())['seeds']
        with np.load(A / pr / 'plans.npz') as z: P = {k: z[k] for k in z.files}
        if not any(k.startswith(names[0]) for k in P): continue
        pn.add_shrinkage(P, names, seeds)
        T = lambda n, c: [P[f'{n}_s{s}_{c}'] for s in seeds]
        sp = lambda c: np.mean([pn.spread(T(n, c)) for n in names])
        A1 = sp('heun32_t1.0')
        print(f'\n== {mk} clips {rng_}')
        print('  spread ratio A_tau/A1:', {t: round(sp(f'heun32_t{t}')/A1, 3) for t in pn.TAUS})
        cfgs = ['euler1_t1.0', 'euler2_t1.0', 'euler4_t1.0', 'euler6_t1.0', 'euler10_t1.0', 'heun3_t1.0', 'heun5_t1.0']
        for c in cfgs:
            if f'{names[0]}_s{seeds[0]}_{c}' not in P: continue
            dist = np.mean([d(P[f'{n}_s{s}_{c}'], P[f'{n}_s{s}_heun32_t1.0']) for n in names for s in seeds])
            print(f'  {c:13s} spread {sp(c):.3f} ratio {sp(c)/A1:.3f}  mean dist to A1 {dist:.3f}')
        if rng_ != (100, 160): continue
        # minADE6 sampler vs reference
        for c in ['euler4_t1.0', 'euler10_t1.0']:
            e = S['models'][mk][c]['minade6']; print(f'  minADE6 {c}: ref {e["reference"]:.3f} sampler {e["sampler"]:.3f} effect {e["effect"][0]:+.3f} [{e["effect"][1]:+.3f},{e["effect"][2]:+.3f}]')
        # per-clip energy distance & decision TV: E4 and paper shrink4 vs bracketing temperatures
        for a, bs in [('euler4_t1.0', ['heun32_t0.6', 'heun32_t0.7']), ('shrink4_t1.0', ['heun32_t0.6', 'heun32_t0.7']),
                      ('euler10_t1.0', ['heun32_t0.8', 'heun32_t0.9']), ('shrink10_t1.0', ['heun32_t0.8', 'heun32_t0.9'])]:
            for b in bs:
                e = [ed(T(n, a), T(n, b)) for n in names]; t = [tv(T(n, a), T(n, b), REC[n]['speed_mps']) for n in names]
                print(f'  per-clip {a:13s} vs {b}: energy {np.mean(e):.3f}  decisionTV {100*np.mean(t):.1f}%')
        for a, b in [('heun32_t0.6', 'heun32_t0.7'), ('heun32_t0.8', 'heun32_t0.9')]:
            e = [ed(T(n, a), T(n, b)) for n in names]; print(f'  per-clip baseline {a} vs {b}: energy {np.mean(e):.3f}')
        # per-noise decision disagreement E4 vs nearest-tau reference, adjacent-tau baseline
        for a, b in [('euler4_t1.0', 'heun32_t0.6'), ('heun32_t0.6', 'heun32_t0.7'), ('euler10_t1.0', 'heun32_t0.8'), ('heun32_t0.8', 'heun32_t0.9'), ('shrink4_t1.0', 'heun32_t0.6')]:
            dis = np.mean([key(decisions(P[f'{n}_s{s}_{a}'], REC[n]['speed_mps'])) != key(decisions(P[f'{n}_s{s}_{b}'], REC[n]['speed_mps'])) for n in names for s in seeds])
            print(f'  per-noise disagreement {a} vs {b}: {100*dis:.1f}%')
# tau_eff vs Gaussian
for mk, _ in MODELS:
    for c, g in [('euler4_t1.0', '4'), ('euler10_t1.0', '10')]:
        t = S['models'][mk][c]['tau_eff']; print(mk, c, 'tau_eff', [round(x, 3) for x in t], 'gauss', round(G['steps'][g]['predicted_median'], 3))
print('stop clips speeds', {n: round(REC[n]['speed_mps'], 2) for n in S['stop_clips']})
print('heun in_range', {c: S['models']['alpamayo15'][c]['in_range'] for c in ['heun3_t1.0', 'heun5_t1.0']})
print('gaussian artifact keys', list(G.keys())[:20])
