"""Exactly solvable decision-commitment model for flow-matching policies.

Data: p(x1)=w N(+mu,s2 I)+(1-w) N(-mu,s2 I); linear interpolation x_t=(1-t)z+t x1.
The decision is the mode reached, sign(mu.x1). The velocity field is closed form,
so we can compare Euler flip rates, windowed freeze/kick susceptibility and the
information rate dI(decision;x_t)/dt without any learned model.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def velocity(x,t,mu,s2,w):
    var=(1-t)**2+t**2*s2;proj=x@mu
    logit=np.log(w/(1-w))+2*t*proj/var;p=1/(1+np.exp(-np.clip(logit,-60,60)))
    gain=(t*s2-(1-t))/var;m=(2*p-1)[:,None]
    return m*mu[None]+gain*(x-t*m*mu[None])


def posterior_plus(x,t,mu,s2,w):
    var=(1-t)**2+t**2*s2;logit=np.log(w/(1-w))+2*t*(x@mu)/var
    return 1/(1+np.exp(-np.clip(logit,-60,60)))


def integrate(z,mu,s2,w,steps,freeze=None,kick=None,direction=None,method='euler'):
    x=z.copy();v=None
    for i in range(steps):
        t=i/steps;dt=1/steps
        if kick is not None and i==kick[0]:x=x+kick[1]*np.linalg.norm(x,axis=1,keepdims=True)*direction
        if not(freeze is not None and freeze[0]<=i<freeze[1] and v is not None):v=velocity(x,t,mu,s2,w)
        if method=='rk4':
            k2=velocity(x+.5*dt*v,t+.5*dt,mu,s2,w);k3=velocity(x+.5*dt*k2,t+.5*dt,mu,s2,w);k4=velocity(x+dt*k3,t+dt,mu,s2,w)
            x=x+dt*(v+2*k2+2*k3+k4)/6
        else:x=x+dt*v
    return x


def binary_entropy(p):
    p=np.clip(p,1e-12,1-1e-12);return -(p*np.log2(p)+(1-p)*np.log2(1-p))


def run(separation,s2,w,dim,n,rng):
    mu=np.zeros(dim);mu[0]=separation;z=rng.standard_normal((n,dim))
    exact=np.sign(integrate(z,mu,s2,w,400,method='rk4')@mu)
    out={'separation':separation,'s2':s2,'w':w,'decision_entropy_bits':float(binary_entropy((exact>0).mean()))}
    out['euler_flip']={k:float((np.sign(integrate(z,mu,s2,w,k)@mu)!=exact).mean()) for k in [2,4,8,16,32]}
    base=np.sign(integrate(z,mu,s2,w,20)@mu);direction=rng.standard_normal(dim);direction/=np.linalg.norm(direction)
    out['freeze_flip']=[float((np.sign(integrate(z,mu,s2,w,20,freeze=(a,a+2))@mu)!=base).mean()) for a in range(0,20,2)]
    out['kick_flip']=[float((np.sign(integrate(z,mu,s2,w,20,kick=(a,.05),direction=direction)@mu)!=base).mean()) for a in range(0,20,2)]
    # Mutual information I(decision; x_t) along the exact flow, decision = exact endpoint mode.
    times=np.linspace(0,1,21);info=[];x=z.copy();prior=binary_entropy((exact>0).mean())
    for i,t in enumerate(times[:-1]):
        info.append(float(prior-binary_entropy(posterior_plus(x,t,mu,s2,w)).mean()))
        x=integrate(x,mu,s2,w,20,method='rk4') if False else x+(1/20)*velocity(x,t,mu,s2,w)
    out['info_bits']=info
    var=lambda t:(1-t)**2+t**2*s2
    grid=np.linspace(1e-3,1-1e-3,2000);out['speciation_time']=float(grid[np.argmin(np.abs(grid*separation/np.sqrt(var(grid))-1))])
    rate=np.diff(info);out['info_rate_peak_time']=float(times[1:-1][np.argmax(rate)]) if len(rate) else None
    out['freeze_peak_time']=float(np.argmax(out['freeze_flip'])*.1);out['kick_peak_time']=float(np.argmax(out['kick_flip'])*.1)
    out['corr_freeze_vs_info_rate']=float(np.corrcoef(out['freeze_flip'],np.add.reduceat(np.r_[rate,0],range(0,20,2))[:10])[0,1]) if np.std(out['freeze_flip'])>0 else None
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    rng=np.random.default_rng(20261022);results=[]
    for separation in [.5,1.,2.,4.]:
        for w in [.5,.8]:
            r=run(separation,.05,w,16,20000,rng);results.append(r)
            print(json.dumps({k:r[k] for k in ['separation','w','decision_entropy_bits','euler_flip','speciation_time','info_rate_peak_time','freeze_peak_time','kick_peak_time','corr_freeze_vs_info_rate']}),flush=True)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps({'scope':__doc__,'results':results},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
