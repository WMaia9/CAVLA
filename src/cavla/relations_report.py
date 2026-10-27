"""Generate static, reproducible plots and tables from completed experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cavla.active_pilot import save_json


def run():
    root = Path('artifacts')
    screening = json.loads((root/'relations_v1_20261020/runs.json').read_text())
    confirmation = json.loads((root/'relations_confirmation_v1_20261020/runs.json').read_text())
    controls = json.loads((root/'relations_controls_v1_20261020/runs.json').read_text())
    utility = json.loads((root/'utility_probe_v1_20261020/runs.json').read_text())
    if (len(screening), len(confirmation), len(controls), len(utility)) != (40, 40, 20, 5):
        raise ValueError('Experiments incomplete')
    runs = screening+confirmation+controls
    methods = ['factual_only', 'factual_3x', 'null_copy', 'weak_null', 'weak_relation',
               'null_copy_relation', 'anchor_relation', 'all_invariance', 'full_cf16', 'full_cf96']
    summary = {}
    for method in methods:
        rows = [r for r in runs if r['method'] == method]
        summary[method] = dict(primary=float(np.mean([r['metrics']['primary'] for r in rows])),
                              training_seconds=float(np.mean([r['training']['seconds'] for r in rows])),
                              **{g: float(np.mean([r['metrics'][g]['action_mse'] for r in rows]))
                                 for g in ('id', 'severity', 'lane_swap', 'composition')})
    save_json(root/'relations_controls_v1_20261020/combined_summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    for i, method in enumerate(methods):
        rows = [r for r in runs if r['method'] == method]
        values = [r['metrics']['primary']*1000 for r in rows]
        axes[0].barh(i, np.mean(values), color='#326fa8' if 'relation' in method else '#999999', alpha=.7)
        axes[0].scatter(values, [i]*len(values), s=15, color='black', alpha=.5)
    axes[0].set_yticks(range(len(methods)), methods); axes[0].invert_yaxis()
    axes[0].set_xlabel('MSE de ação × 1.000; menor é melhor')
    axes[0].set_title('Triagem + confirmação (10 seeds)\nControles adicionais exploratórios')
    for method in ('factual_only', 'null_copy', 'null_copy_relation', 'anchor_relation', 'full_cf96'):
        axes[1].plot(range(4), [summary[method][g] for g in ('id', 'severity', 'lane_swap', 'composition')], 'o-', label=method)
    axes[1].set_xticks(range(4), ['ID', 'Severidade', 'Troca de faixa', 'Composição'], rotation=15)
    axes[1].set_yscale('log'); axes[1].set_ylabel('MSE de ação; escala log')
    axes[1].set_title('A média primária é dominada pela severidade')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.suptitle('Supervisão parcial: ambiente sintético, CNN, sem VLA real', fontsize=13)
    plot = Path('docs/plots/relations_v1.png'); fig.savefig(plot, dpi=180); plt.close(fig)
    proxies = list(utility[0]['proxies'])
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for i, proxy in enumerate(proxies):
        values = [r['proxies'][proxy]['spearman'] for r in utility]
        ax.bar(i, np.mean(values), color='#326fa8', alpha=.65)
        ax.scatter([i]*len(values), values, color='black', s=25)
    ax.axhline(0, color='gray', linewidth=.6)
    ax.set_xticks(range(len(proxies)), ['Erro de ação', 'Erro de efeito', 'Violação de relação', 'Alinhamento de gradiente'])
    ax.set_ylabel('Spearman com ganho real na calibração')
    ax.set_title('240 micro-adaptações: consultar todos os rótulos não é aquisição sem rótulos')
    fig.savefig('docs/plots/utility_probe_v1.png', dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': run()
