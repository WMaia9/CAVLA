"""Per-sample contraction signature (proposal, presentation review round 5) -> fig_signature.pdf.

Finding 4 (find:shrink) states that rate-level matching cannot tell E_k from a spread-matched shrinkage,
but the per-sample contraction ratio |S - E_1| / |A_1 - E_1| (same noise) can, and that Euler's
distribution of ratios resembles the reference solver's at the matched temperature. The paper gives
only medians and fractions; this figure shows the three distributions.

  * Euler E_k: the per-sample ratios (6 seeds x 60 clips per model, three models pooled).
  * A_tau_eff: the reference solver at E_k's effective temperature. Its ratios are obtained by
    interpolating the quantile function between the two tau-grid points that bracket E_k's mean spread,
    with the same linear weight cavla.paper_numbers uses for matched_median (at the median the two coincide).
  * Shrink_k: E_1 + c (A_1 - E_1) with one global c per model, so every sample has ratio exactly c
    (drawn as a vertical step at c). This is a property of the control's construction, not a finding.

Also prints clip-bootstrap intervals (clips resampled jointly across models, 2000 draws) for the
median and fraction-contracted differences Euler - A_tau_eff, so the text can quote them.
Numbers are consistent with artifacts/paper_numbers_v3_20261026/summary.json (contraction.median,
matched_median); the script checks that and fails loudly if they disagree.

Run from paper/:  PYTHONPATH=../src <python> figures/make_fig_signature.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

fs.NUMBERS = fs.ART / 'paper_numbers_v3_20261026' / 'summary.json'
MODELS = ['alpamayo15', 'alpamayo_r1', 'alpamayo2']
CURVE = [f'heun32_t{t}' for t in fs.TAUS]


def ratios(table, names, config):
    """Per-clip arrays of per-sample ratios: shape (clips, seeds)."""
    return np.array([table[config][n]['contraction'] for n in names])


def matched_weights(table, names, sampler):
    spreads = np.array([np.mean([table[c][n]['spread'] for n in names]) for c in CURVE])
    s = np.mean([table[sampler][n]['spread'] for n in names])
    o = np.argsort(spreads); sp = spreads[o]; j = int(np.clip(np.searchsorted(sp, s), 1, len(sp) - 1))
    w = (s - sp[j - 1]) / (sp[j] - sp[j - 1])
    return CURVE[o[j - 1]], CURVE[o[j]], float(w)


def interp_quantiles(a, b, w, q):
    return (1 - w) * np.quantile(a, q) + w * np.quantile(b, q)


def main():
    fs.style()
    names, T = fs.tables(); num = fs.numbers()['models']
    q = np.linspace(0.0005, 0.9995, 2000)
    rng = np.random.default_rng(fs.BOOT_SEED); idx = [rng.integers(0, len(names), len(names)) for _ in range(2000)]
    fig, axes = plt.subplots(1, 2, figsize=(fs.COL, 1.55), sharey=True, gridspec_kw=dict(wspace=0.12))
    report = {}
    for ax, k, letter in zip(axes, (4, 10), 'ab'):
        E, A0, A1, W, C = [], [], [], [], []
        for model in MODELS:
            t = T[model]; e = ratios(t, names, f'euler{k}_t1.0'); lo, hi, w = matched_weights(t, names, f'euler{k}_t1.0')
            a0, a1 = ratios(t, names, lo), ratios(t, names, hi); sh = ratios(t, names, f'shrink{k}_t1.0')
            # consistency with the summary the tables read
            ref = num[model][f'euler{k}_t1.0']['contraction']
            assert abs(np.median(e) - ref['median']) < 1e-6, (model, k, 'euler median')
            assert abs(interp_quantiles(a0, a1, w, .5) - ref['matched_median']) < 0.02, (model, k, 'matched median')
            assert np.ptp(sh) < 1e-3, 'shrink ratio should be constant (one global c per model)'
            E.append(e); A0.append(a0); A1.append(a1); W.append(w); C.append(float(sh.flat[0]))
        def stats(sel):
            e = np.concatenate([E[m][sel].ravel() for m in range(3)])
            a = np.concatenate([interp_quantiles(A0[m][sel].ravel(), A1[m][sel].ravel(), W[m], q) for m in range(3)])
            # a is a quantile sample per model (2000 each), so medians/fractions are of the pooled mixture
            return np.median(e) - np.median(a), np.mean(e < 1) - np.mean(a < 1), np.median(e), np.median(a), np.mean(e < 1), np.mean(a < 1)
        point = stats(slice(None)); boots = np.array([stats(i) for i in idx])
        ci = lambda j: np.quantile(boots[:, j], [.025, .975]).tolist()
        report[f'E{k}'] = {'median_euler': point[2], 'median_temp': point[3], 'median_diff': [point[0], *ci(0)],
                           'frac_euler': point[4], 'frac_temp': point[5], 'frac_diff': [point[1], *ci(1)], 'shrink_c': C,
                           'n_samples_per_curve': int(sum(x.size for x in E)), 'n_clips': len(names)}
        e = np.sort(np.concatenate([x.ravel() for x in E]))
        a = np.sort(np.concatenate([interp_quantiles(A0[m].ravel(), A1[m].ravel(), W[m], q) for m in range(3)]))
        ax.axvline(1, color=fs.GRAY, lw=0.4, ls=':', zorder=0)
        ax.plot(a, np.linspace(0, 1, len(a)), color=fs.BLACK, lw=1.0, label=r'reference at $\tau_{\mathrm{eff}}$')
        ax.plot(e, np.linspace(0, 1, len(e)), color=fs.BLUE, lw=1.0, label=fr'Euler $E_k$')
        c = float(np.mean(C)); ax.plot([0, c, c, 2], [0, 0, 1, 1], color=fs.GRAY, lw=0.9, ls='--', zorder=1, label=r'Shrink$_k$ (one factor)')
        ax.set_xlim(0, 1.6); ax.set_ylim(0, 1.02); ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.text(0.98, 0.04, fr"at $E_{{{k}}}$'s spread", transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.set_xlabel(r'$\|S-E_1\|\,/\,\|A_1-E_1\|$')
        fs.panel_letter(ax, letter, x=0.0, y=1.02)
    axes[0].set_ylabel('share of samples')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=3, handlelength=1.4, columnspacing=0.8, fontsize=6.3)
    fs.save(fig, 'fig_signature.pdf')
    print(json.dumps(report, indent=1, default=float))
    # Macros for the text (proposal file; port into src/cavla/paper_numbers.py so they land in summary.json).
    lines = ['% Generated by figures/make_fig_signature.py (proposal); do not edit by hand.']
    for k, tag in ((4, 'Four'), (10, 'Ten')):
        r = report[f'E{k}']; f = [100 * x for x in r['frac_diff']]; m = [0.0 if abs(x) < 0.005 else x for x in r['median_diff']]
        lines += [f'\\newcommand{{\\SigFracDiff{tag}}}{{{f[0]:.1f}}}', f'\\newcommand{{\\SigFracDiff{tag}Lo}}{{{f[1]:.1f}}}',
                  f'\\newcommand{{\\SigFracDiff{tag}Hi}}{{{f[2]:.1f}}}',
                  f'\\newcommand{{\\SigMedDiff{tag}}}{{{m[0]:.2f}}}'.replace('-', '$-$'),
                  f'\\newcommand{{\\SigMedDiff{tag}Lo}}{{{m[1]:.2f}}}'.replace('-', '$-$'),
                  f'\\newcommand{{\\SigMedDiff{tag}Hi}}{{{m[2]:.2f}}}'.replace('-', '$-$')]
    (fs.OUT / 'sig_numbers_v5_proposal.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
