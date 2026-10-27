"""Decision-information schedule versus entropic time (Stancevic et al., NeurIPS 2025) in the solvable model.

Entropic time spaces steps by the conditional entropy rate of the whole output,
H(x1|x_t) = E[H(mode|x_t)] + d/2 log(2 pi e sigma_post^2(t)). The decision schedule
uses only the information about the decision, I(mode_final; x_t). Both are
estimated on an independent pilot batch and compared at equal field evaluations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cavla.commitment_theory import velocity, posterior_plus, integrate, binary_entropy
from cavla.commitment_theory_v2 import integrate_grid, cached_middle, grid_from_density


def curves(z,mu,s2,w,dim,points=200):
    final=(integrate(z,mu,s2,w,400,method='rk4')@mu>0).mean();prior=binary_entropy(final)
    times=np.linspace(0,1,points+1)[:-1];x=z.copy();decision=[];total=[]
    for t in times:
        var=(1-t)**2+t**2*s2;post_var=s2*(1-t)**2/var
        mode_entropy=binary_entropy(posterior_plus(x,t,mu,s2,w)).mean()
        decision.append(prior-mode_entropy)
        total.append(mode_entropy*np.log(2)+.5*dim*np.log(2*np.pi*np.e*max(post_var,1e-300)))
        x=x+velocity(x,t,mu,s2,w)/points
    return times,np.array(decision),np.array(total)


def run(separation,s2,w,dim,n,rng):
    mu=np.zeros(dim);mu[0]=separation;z=rng.standard_normal((n,dim));pilot=rng.standard_normal((n//4,dim))
    exact=np.sign(integrate(z,mu,s2,w,400,method='rk4')@mu);flip=lambda x:float((np.sign(x@mu)!=exact).mean())
    times,info,entropy=curves(pilot,mu,s2,w,dim)
    info_rate=np.maximum(np.gradient(info,times),0);entropy_rate=np.maximum(-np.gradient(entropy,times),0)
    # Entropic time cannot place steps exactly at t=1 where the rate diverges; clip the last bins as in practice.
    entropy_rate=np.minimum(entropy_rate,np.quantile(entropy_rate,.98))
    # Rescaled entropic time used for their reported results: rate sigma(t)*|dH/dt|, with data-scale noise
    # sigma(t)=(1-t)/t for x_t=(1-t)z+t x1 (data at t=1 here, noise at t=0).
    rescaled_rate=np.maximum(-np.gradient(entropy,times),0)*(1-times)/np.maximum(times,1e-3)
    out={'separation':separation,'w':w,'dim':dim,'s2':s2,'info_peak':float(times[np.argmax(info_rate)]),
        'entropy_rate_median_time':float(np.interp(.5,np.cumsum(entropy_rate)/entropy_rate.sum(),times)),
        'rescaled_median_time':float(np.interp(.5,np.cumsum(rescaled_rate)/rescaled_rate.sum(),times)),'by_nfe':{}}
    for k in [3,4,6,8]:
        out['by_nfe'][k]={'uniform':flip(integrate(z,mu,s2,w,k)),
            'entropic_time':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,entropy_rate+1e-12,k))),
            'rescaled_entropic_time':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,rescaled_rate+1e-12,k))),
            'decision_info':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,info_rate+1e-3*info_rate.max(),k))),
            'decision_info_plus_uniform':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,info_rate/info_rate.mean()+1,k)))}
        if k==4:out['by_nfe'][k]['flashdrive_cached8']=flip(cached_middle(z,mu,s2,w,8))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    rng=np.random.default_rng(20261024);results=[]
    for dim in [2,16,64]:
        for separation in [1.,2.,4.]:
            for w in [.65,.8]:
                r=run(separation,.05,w,dim,20000,rng);results.append(r);print(json.dumps(r),flush=True)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'results':results},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    methods=['uniform','entropic_time','rescaled_entropic_time','decision_info','decision_info_plus_uniform']
    cases=[(r,k) for r in results for k in r['by_nfe']]
    best={m:sum(min(r['by_nfe'][k][o] for o in methods)==r['by_nfe'][k][m] for r,k in cases) for m in methods}
    mean={m:float(np.mean([r['by_nfe'][k][m] for r,k in cases])) for m in methods}
    beats={m:sum(r['by_nfe'][k]['decision_info_plus_uniform']<r['by_nfe'][k][m] for r,k in cases) for m in methods}
    print(json.dumps({'cases':len(cases),'times_best':best,'mean_flip':mean,'decision_info_plus_uniform_beats':beats}))


if __name__=='__main__':main()
