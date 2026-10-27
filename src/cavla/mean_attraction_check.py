"""Where do numerically flipped decisions go? Solvable K-mode Gaussian-mixture flows.

For every sample whose class under an under-resolved sampler differs from the exact flow,
record whether it lands in the class containing the mixture mean, in the most probable
class, and whether its endpoint is closer to the global mean than the exact endpoint.
Uses the same configurations and seed as mode_collapse_theory.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cavla.mode_collapse_theory import classify, solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    rng=np.random.default_rng(31);rows=[]
    for k in [2,3,5]:
        for dim in [2,16]:
            for scale in [1.,3.]:
                for trial in range(3):
                    weights=rng.dirichlet(np.ones(k)*1.5);means=rng.standard_normal((k,dim))*scale;z=rng.standard_normal((20000,dim))
                    exact_x=solve(z,means,weights,.05,np.linspace(0,1,401),'rk4');exact=classify(exact_x,means)
                    global_mean=(weights[:,None]*means).sum(0);mean_class=int(classify(global_mean[None],means)[0])
                    for method,steps in [('euler',2),('euler',4),('cached',8)]:
                        x=solve(z,means,weights,.05,np.linspace(0,1,steps+1),method);c=classify(x,means);flip=c!=exact
                        if not flip.any():continue
                        closer=np.linalg.norm(x[flip]-global_mean,axis=1)<np.linalg.norm(exact_x[flip]-global_mean,axis=1)
                        rows.append({'k':k,'dim':dim,'scale':scale,'trial':trial,'sampler':f'{method}{steps}','flips':int(flip.sum()),
                            'to_mean_class':int((c[flip]==mean_class).sum()),'to_most_probable':int((c[flip]==int(np.argmax(weights))).sum()),
                            'closer_to_global_mean':int(closer.sum())})
    total=sum(r['flips'] for r in rows)
    summary={'flips':total,'fraction_to_mean_class':sum(r['to_mean_class'] for r in rows)/total,
        'fraction_to_most_probable':sum(r['to_most_probable'] for r in rows)/total,
        'fraction_closer_to_global_mean':sum(r['closer_to_global_mean'] for r in rows)/total}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'summary':summary,'rows':rows},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=1))


if __name__=='__main__':main()
