"""Standalone plots for the restricted response-coding investigation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rate',type=Path,default=Path('artifacts/maxplus_rate_probe_v1_20261022/summary.json'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    records=json.loads(args.rate.read_text())['records'];rows=[r for r in records if r['nodes']==256]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    ax=axes[0];n=64;k=np.arange(n-1);m=(k+1)*n-(k+1)*(k+2)/2
    ax.plot(k,m,color='#28689b',label='Resposta exata a até k remoções')
    ax.axhline(n-1,color='#888888',linestyle='--',label='Somente factual')
    ax.set(xlabel='Máximo de vértices removidos (k)',ylabel='Coordenadas independentes necessárias',title='Classe restrita, n=64');ax.legend(fontsize=8)
    ax=axes[1];x=[r['order']+1 for r in rows]
    for method,label,color in [('uniform_potential','Grade uniforme','#888888'),('adaptive_potential','Precisão variável, potenciais','#28689b'),('adaptive_chain','Precisão variável, feedback clássico','#c7771d')]:
        ax.plot(x,[r['methods'][method]['payload_bits']/1000 for r in rows],marker='o',label=label,color=color)
    ax.plot(x,[r['covering_lower_bound_bits']/1000 for r in rows],linestyle='--',label='Limite inferior de cobertura',color='black')
    ax.set(xscale='log',xlabel='k+1 (escala log)',ylabel='Payload / 1.000 bits',title='n=256, erro uniforme garantido ≤0,02');ax.legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.suptitle('Grafos conhecidos de precedência: não é compressão de VLA',fontsize=12)
    fig.savefig(args.out/'capacity_and_bits.png',dpi=180);fig.savefig(args.out/'capacity_and_bits.pdf');plt.close(fig)


if __name__=='__main__':main()
