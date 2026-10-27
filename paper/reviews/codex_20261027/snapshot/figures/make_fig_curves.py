"""Main result figure (figure*, text width).

Columns: the three Alpamayo generations. Rows: single-sample error ADE_1, decision flips relative
to A_1 (same noise), best-of-six error minADE_6. Black curve with gray band: the reference solver
(Heun-32) started from tau*z, 95% clip-bootstrap band. Blue markers: Euler samplers E_k placed at
their effective temperature (Eq. 2; horizontal bar = 95% interval of tau_eff). Vermillion: Heun-3
and Heun-5 (Alpamayo-1.5 only). A thin gray tick joins each marker to the curve at the same tau:
its length is the residual M(S) - M(A_tau_eff) reported in Table 1. Samplers whose tau_eff falls
below the measured range (tau < 0.4: E_1 everywhere, E_2 for R1 and Alpamayo-2) are not drawn.

Run from paper/:  PYTHONPATH=../src <python> figures/make_fig_curves.py [--rows ade1,flip,minade6]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

MODELS = ['alpamayo15', 'alpamayo_r1', 'alpamayo2']
ROWS = {'ade1': (r'$\Delta$ADE$_1$ vs. $A_1$ (m)', 1.), 'flip': (r'decisions changed (%)', 100.),
        'minade6': (r'$\Delta$minADE$_6$ vs. $A_1$ (m)', 1.)}
LABEL_OFFSET = {2: (0, 5), 4: (0, 5), 6: (0, 5), 10: (0, 5)}


def draw(ax, table, names, samples, num, model, metric, scale, label_euler, label_heun):
    tau, y, lo, hi = fs.curve(table, names, metric, samples, scale, paired=True)
    ax.axhline(0, color=fs.GRAY, lw=0.4, ls=':', zorder=0)
    ax.fill_between(tau, lo, hi, color=fs.BAND, alpha=0.6, lw=0, zorder=1)
    ax.plot(tau, y, color=fs.BLACK, lw=1.0, zorder=2)
    samplers = [(f'euler{k}_t1.0', f'$E_{{{k}}}$', fs.BLUE, 'o', k) for k in (2, 4, 6, 10)]
    if model == 'alpamayo15':
        samplers += [(f'heun{k}_t1.0', f'Heun-{k}', fs.VERMILLION, 'D', None) for k in (3, 5)]
    for config, name, color, marker, k in samplers:
        e = num.get(config)
        if e is None or not e['in_range']:
            continue
        ref = e[metric]['reference'];x = e['tau_eff']
        v = (e[metric]['sampler'] - ref) * scale;m = (e[metric]['matched'] - ref) * scale
        ax.plot([x[0], x[0]], [m, v], color=fs.GRAY, lw=0.8, zorder=3, solid_capstyle='butt')
        ax.errorbar(x[0], v, xerr=[[x[0] - x[1]], [x[2] - x[0]]], fmt=marker, ms=3.4 if marker == 'o' else 3.0,
                    color=color, mec='white', mew=0.4, elinewidth=0.6, capsize=0, zorder=4)
        if (k is not None and label_euler) or (k is None and label_heun):
            off = LABEL_OFFSET.get(k, (-4, 3));ha = 'center' if off[0] == 0 else ('left' if off[0] > 0 else 'right')
            ax.annotate(name, (x[0], v), xytext=off, textcoords='offset points', ha=ha, va='bottom', fontsize=6.5)
    ax.set_xlim(0.37, 1.03);ax.set_xticks([0.4, 0.6, 0.8, 1.0])


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--rows', default='ade1,flip,minade6')
    p.add_argument('--out', default='fig_curves.pdf');args = p.parse_args()
    rows = args.rows.split(',')
    fs.style()
    names, T = fs.tables();samples = fs.boot_samples(names);num = fs.numbers()['models']
    fig, axes = plt.subplots(len(rows), 3, figsize=(fs.TEXT, 1.25 * len(rows) + 0.35), sharex=True, sharey='row',
                             gridspec_kw=dict(hspace=0.22, wspace=0.12))
    axes = np.atleast_2d(axes)
    for j, model in enumerate(MODELS):
        axes[0, j].set_title(fs.LABEL[model], fontsize=8)
        for i, metric in enumerate(rows):
            label, scale = ROWS[metric]
            draw(axes[i, j], T[model], names, samples, num[model], model, metric, scale,
                 label_euler=(i == 0), label_heun=(metric == 'flip'))
            if j == 0:
                axes[i, j].set_ylabel(label)
        axes[-1, j].set_xlabel(r'temperature $\tau$ or $\tau_{\mathrm{eff}}$')
    for i in range(len(rows)):
        fs.panel_letter(axes[i, 0], 'abcdefg'[i], x=-0.26, y=0.98)
    axes[[r for r in rows].index('flip') if 'flip' in rows else 0, 0].set_ylim(bottom=0)
    h = [plt.Line2D([], [], color=fs.BLACK, lw=1.0, label=r'reference $A_\tau$ (Heun-32 from $\tau z$), 95% band'),
         plt.Line2D([], [], color=fs.BLUE, lw=0, marker='o', ms=3.4, label=r'Euler $E_k$ at $\tau_{\mathrm{eff}}$'),
         plt.Line2D([], [], color=fs.VERMILLION, lw=0, marker='D', ms=3, label=r'Heun-$k$ at $\tau_{\mathrm{eff}}$'),
         plt.Line2D([], [], color=fs.GRAY, lw=0.8, label='residual to the curve')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.94 if len(rows) > 2 else 0.975), ncol=4, handlelength=1.6, columnspacing=1.4)
    fs.save(fig, args.out)
    for model in MODELS:
        for c, e in num[model].items():
            if e['in_range'] and c.endswith('_t1.0'):
                print(f"{model:11s} {c:13s} tau {e['tau_eff'][0]:.2f} " + ' '.join(
                    f"{m} d{(e[m]['sampler'] - e[m]['reference']) * ROWS[m][1]:+.2f}/res{(e[m]['sampler'] - e[m]['matched']) * ROWS[m][1]:+.2f}" for m in rows))


if __name__ == '__main__':
    main()
