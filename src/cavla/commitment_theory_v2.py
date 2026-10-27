"""Solvable-model tests v2: order scaling, normalized kicks, and information-rate schedules.

Same two-mode Gaussian flow as commitment_theory. Schedules are compared at equal
field evaluations against the exact decision: uniform Euler, FlashDrive-style
cached middle, a cosine-variation schedule (steps where consecutive velocities
turn), and an information-rate schedule (step density proportional to dI/dt,
estimated on an independent pilot batch).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cavla.commitment_theory import velocity, posterior_plus, integrate, binary_entropy


def integrate_grid(z,mu,s2,w,grid,method='euler'):
    x=z.copy()
    for a,b in zip(grid[:-1],grid[1:]):
        v=velocity(x,a,mu,s2,w)
        if method=='heun':x=x+.5*(b-a)*(v+velocity(x+(b-a)*v,b,mu,s2,w))
        else:x=x+(b-a)*v
    return x


def cached_middle(z,mu,s2,w,steps):
    x=z.copy();v=None
    for i in range(steps):
        t=i/steps
        if not(2<=i<steps-2):v=velocity(x,t,mu,s2,w)
        x=x+v/steps
    return x


def grid_from_density(times,density,k):
    cdf=np.r_[0,np.cumsum(np.maximum(density,1e-9))];cdf/=cdf[-1];edges=np.linspace(0,1,len(cdf))
    grid=np.interp(np.linspace(0,1,k+1),cdf,edges);grid[0],grid[-1]=0.,1.;return grid


def information_curve(z,mu,s2,w,points=50):
    exact_plus=(integrate(z,mu,s2,w,400,method='rk4')@mu>0).mean();prior=binary_entropy(exact_plus)
    times=np.linspace(0,1,points+1);x=z.copy();info=[]
    for i,t in enumerate(times[:-1]):
        info.append(prior-binary_entropy(posterior_plus(x,t,mu,s2,w)).mean());x=x+(1/points)*velocity(x,t,mu,s2,w)
    return times[:-1],np.array(info)


def cosine_curve(z,mu,s2,w,points=50):
    times=np.linspace(0,1,points+1);x=z.copy();prev=None;turn=[]
    for t in times[:-1]:
        v=velocity(x,t,mu,s2,w)
        if prev is not None:turn.append(1-np.mean(np.sum(v*prev,1)/(np.linalg.norm(v,axis=1)*np.linalg.norm(prev,axis=1))))
        else:turn.append(0.)
        prev=v;x=x+v/points
    return np.array(turn)


def run(separation,s2,w,dim,n,rng):
    mu=np.zeros(dim);mu[0]=separation;z=rng.standard_normal((n,dim));pilot=rng.standard_normal((n//4,dim))
    exact=np.sign(integrate(z,mu,s2,w,400,method='rk4')@mu)
    flip=lambda x:float((np.sign(x@mu)!=exact).mean())
    out={'separation':separation,'w':w}
    out['euler']={k:flip(integrate(z,mu,s2,w,k)) for k in [4,8,16,32]}
    out['heun_nfe']={2*k:flip(integrate_grid(z,mu,s2,w,np.linspace(0,1,k+1),'heun')) for k in [2,4,8,16]}
    # Kick scaled by the conditional standard deviation at that time, not by |x|.
    base=np.sign(integrate(z,mu,s2,w,20)@mu);direction=rng.standard_normal(dim);direction/=np.linalg.norm(direction)
    kicks=[]
    for a in range(20):
        t=a/20;x=z.copy()
        for i in range(20):
            if i==a:x=x+.05*np.sqrt((1-t)**2+t**2*s2)*direction
            x=x+velocity(x,i/20,mu,s2,w)/20
        kicks.append(float((np.sign(x@mu)!=base).mean()))
    times,info=information_curve(pilot,mu,s2,w);rate=np.maximum(np.gradient(info,times),0);turn=cosine_curve(pilot,mu,s2,w)
    out['kick_peak_time']=float(np.argmax(kicks)/20);out['info_rate_peak_time']=float(times[np.argmax(rate)])
    out['schedules']={}
    for k in [4,6,8]:
        out['schedules'][k]={'uniform':flip(integrate(z,mu,s2,w,k)),
            'cached_middle_8_as_4nfe':flip(cached_middle(z,mu,s2,w,8)) if k==4 else None,
            'cosine_turn':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,turn+1e-3,k))),
            'info_rate':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,rate+1e-3*rate.max(),k))),
            'info_rate_plus_uniform':flip(integrate_grid(z,mu,s2,w,grid_from_density(times,rate/rate.mean()+1.,k)))}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    rng=np.random.default_rng(20261023);results=[]
    for separation in [.5,1.,2.,4.]:
        for w in [.65,.8,.95]:
            r=run(separation,.05,w,16,20000,rng);results.append(r);print(json.dumps(r),flush=True)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'results':results},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
