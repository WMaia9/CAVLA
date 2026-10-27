"""Shared style and data access for the paper figures.

Every number comes from the artifacts the paper already uses:
  - per-clip plans: artifacts/temperature_probe{,_r1,_a2}_v1_20261022/plans.npz
  - matched-temperature point estimates and intervals: artifacts/paper_numbers_v4_20261026/summary.json
  - Gaussian contraction prediction: artifacts/gaussian_contraction_v1_20261026/summary.json
Per-clip metrics (spread, ADE1, minADE6, flips) are recomputed with cavla.paper_numbers, the same
code that generated summary.json, on the same 60 clips (context_100-159) and the same reference.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts'
OUT = Path(__file__).resolve().parent
DATA = ART / 'physicalai_test_contexts_v1_20261022'
NUMBERS = ART / 'paper_numbers_v4_20261026' / 'summary.json'
GAUSS = ART / 'gaussian_contraction_v1_20261026' / 'summary.json'
PROBES = {'alpamayo15': 'temperature_probe_v1_20261022', 'alpamayo_r1': 'temperature_probe_r1_v1_20261022',
          'alpamayo2': 'temperature_probe_a2_v1_20261022'}
LABEL = {'alpamayo15': 'Alpamayo-1.5', 'alpamayo_r1': 'Alpamayo-R1', 'alpamayo2': 'Alpamayo-2'}
MARKER = {'alpamayo15': 'o', 'alpamayo_r1': 's', 'alpamayo2': '^'}
TAUS = [1., .9, .8, .7, .6, .5, .4]
SEEDS = [9201, 9202, 9203, 9204, 9205, 9206]
BOOT, BOOT_SEED = 2000, 20261026

# Okabe-Ito. Reference / explicit temperature = black with a gray band; Euler = blue;
# second-order (Heun) = vermillion; human log = gray dashed.
BLACK, BLUE, SKY, VERMILLION, GREEN, ORANGE = '#000000', '#0072B2', '#56B4E9', '#D55E00', '#009E73', '#E69F00'
BAND, GRAY = '#BBBBBB', '#777777'
# one-hue ramp for Euler step counts (darker = fewer steps = more contraction)
EULER = {1: '#003A5C', 2: '#004F7D', 4: '#0072B2', 6: '#3F95C6', 10: '#7FB8DA'}

COL, TEXT = 3.25, 6.75  # ICML single-column and text widths (inches)


def style():
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0, 'lines.markersize': 4,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'figure.dpi': 150})


def panel_letter(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontweight='bold', va='bottom', ha='right')


def numbers():
    return json.loads(NUMBERS.read_text())


def gauss():
    return json.loads(GAUSS.read_text())


def plans(model):
    with np.load(ART / PROBES[model] / 'plans.npz') as z:
        return {k: z[k] for k in z.files}


def tables():
    """Per-clip metrics for every config and model, computed by the paper's own generator."""
    from cavla import paper_numbers as pn
    args = argparse.Namespace(data=DATA, artifacts=ART, range=[100, 160])
    return pn.load(args)


def boot_samples(names):
    rng = np.random.default_rng(BOOT_SEED)
    return [list(rng.choice(names, len(names))) for _ in range(BOOT)]


def curve(table, names, metric, samples, scale=1., paired=False):
    """Explicit-temperature curve A_tau: point estimate and 95% clip-bootstrap band.
    paired=True plots the per-clip change relative to A_1, so the band reflects the paired comparison
    rather than between-clip differences in difficulty."""
    configs = [f'heun32_t{t}' for t in TAUS]
    base = np.array([table['heun32_t1.0'][n][metric] for n in names]) if paired else 0.
    arr = {c: np.array([table[c][n][metric] for n in names]) - base for c in configs}
    idx = {n: i for i, n in enumerate(names)}
    point = np.array([arr[c].mean() for c in configs]) * scale
    boots = np.array([[arr[c][[idx[n] for n in s]].mean() for c in configs] for s in samples]) * scale
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    return np.array(TAUS), point, lo, hi


def sampler_ci(table, names, config, metric, samples, scale=1.):
    v = np.array([table[config][n][metric] for n in names]);idx = {n: i for i, n in enumerate(names)}
    b = np.array([v[[idx[n] for n in s]].mean() for s in samples])
    return v.mean() * scale, *(np.quantile(b, [.025, .975]) * scale)


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    print('wrote', path)
