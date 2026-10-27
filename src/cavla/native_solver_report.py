"""Summarize a selected diagnostic without turning strong error into driving risk."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(root):
    with np.load(root/'predictions.npz') as z:return {key:z[key] for key in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    first=Path('artifacts/native_solver_probe_v2_20261022');controls=Path('artifacts/native_solver_classical_controls_v1_20261022')
    bfroot=Path('artifacts/native_solver_convergence_bf16_v1_20261022');fproot=Path('artifacts/native_solver_convergence_fp32_v1_20261022')
    predictions={**load(first),**load(controls)};b=load(bfroot);f=load(fproot)
    runs=json.loads((first/'records.json').read_text())+json.loads((controls/'records.json').read_text())
    nfe={(r['method']+str(r['steps'])):r['field_evaluations'] for r in runs};rows=[];checks=[]
    for context in ['context_000','context_001']:
        prefix=f'{context}_s9202_';ref=b[prefix+'heun160'];vf=predictions[prefix+'euler10_velocities'].reshape(10,-1)
        cos=np.sum(vf[1:]*vf[:-1],axis=1)/(np.linalg.norm(vf[1:],axis=1)*np.linalg.norm(vf[:-1],axis=1))
        checks.append({'context':context,'diagnostic_seed':9202,'initial_velocity_cosine':float(cos[0]),
            'minimum_velocity_cosine':float(cos.min()),
            'bf16_heun80_vs_160_mean_xy_m':float(np.linalg.norm(b[prefix+'heun80'][:,:2]-ref[:,:2],axis=-1).mean()),
            'fp32_heun80_vs_160_mean_xy_m':float(np.linalg.norm(f[prefix+'heun80'][:,:2]-f[prefix+'heun160'][:,:2],axis=-1).mean()),
            'cross_precision_heun160_mean_xy_m':float(np.linalg.norm(f[prefix+'heun160'][:,:2]-ref[:,:2],axis=-1).mean())})
        for key,value in predictions.items():
            if not key.startswith(prefix) or value.shape!=(64,3):continue
            solver=key[len(prefix):];error=np.linalg.norm(value[:,:2]-ref[:,:2],axis=-1)
            rows.append({'context':context,'seed':9202,'solver':solver,'field_evaluations':nfe[solver],
                'mean_xy_m':float(error.mean()),'first_half_second_mean_xy_m':float(error[:5].mean()),
                'first_second_mean_xy_m':float(error[:10].mean()),'terminal_xy_m':float(error[-1]),
                'terminal_delta_x_m':float(value[-1,0]-ref[-1,0]),'terminal_delta_y_m':float(value[-1,1]-ref[-1,1])})
    result={'scope':'Selected same-noise numerical diagnostic, not policy success or safety','rows':rows,'convergence_checks':checks,
        'limitations':['Seed9202 selected after first two-seed results','Two windows from one route, not independent scenes',
            'Heun160 approximate reference has remaining discretization error','Strong pathwise discrepancies do not establish differences of marginal action distributions',
            'Prefix position error does not certify control jerk, safety or closed-loop performance',
            'Timing includes CPU diagnostic copies; concurrent convergence jobs are not a latency benchmark',
            'No learned adaptive algorithm evaluated; classical solvers explain much of the improvement']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    styles={'euler':('#888888','Euler'),'heun':('#28689b','Heun'),'rk4':('#a65628','RK4'),'early_euler':('#5e9460','Euler, grade inicial densa')}
    for ax,context in zip(axes,['context_000','context_001']):
        for method,(color,label) in styles.items():
            subset=[r for r in rows if r['context']==context and r['solver'].rstrip('0123456789')==method]
            subset=sorted(subset,key=lambda r:r['field_evaluations'])
            ax.plot([r['field_evaluations'] for r in subset],[r['mean_xy_m'] for r in subset],marker='o',color=color,label=label)
        ax.set(xscale='log',yscale='log',xlabel='Avaliações do campo',ylabel='Desvio médio XY (m)',title=context)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.suptitle('Diagnóstico selecionado: referência numérica Heun160, não ação correta',fontsize=11)
    fig.savefig(args.out/'classical_controls.png',dpi=180);fig.savefig(args.out/'classical_controls.pdf');plt.close(fig)


if __name__=='__main__':main()
