"""Stable OR and exact partial-observation likelihood baselines."""
import torch
from torch import nn
from torch.nn import functional as F

from cavla.structured_geometry_learning import GeometryHead


def log_or(logits,keep):
    """Log P(any success) via disjoint first-success events, stable at -1000."""
    survival=F.logsigmoid(-logits)*keep
    prefix=torch.cumsum(survival,dim=-1)-survival
    events=(F.logsigmoid(logits)+prefix).masked_fill(~keep,-torch.inf)
    # Empty sets have a fixed finite log probability for numerical training.
    empty=~keep.any(-1)
    events=events.masked_fill(empty[...,None],-1000.)
    result=torch.logsumexp(events,dim=-1)
    return torch.where(empty,torch.full_like(result,-1000.),result)


def or_logits(logits,keep):
    log_positive=log_or(logits,keep)
    log_negative=(F.logsigmoid(-logits)*keep).sum(-1)
    return log_positive-log_negative


def joint_pair_nll(logits,present,removed,observed):
    """Exact NLL of original and one deletion outcome, with latent Bernoullis."""
    others=present.clone();others.scatter_(1,removed[:,None],False)
    log_negative=F.logsigmoid(-logits)
    all_zero=-(log_negative*present).sum(-1)
    unique=-F.logsigmoid(logits.gather(1,removed[:,None]).squeeze(1))-(log_negative*others).sum(-1)
    redundant=-log_or(logits,others)
    result=torch.where(~observed[:,0],all_zero,torch.where(~observed[:,1],unique,redundant))
    impossible=(~observed[:,0])&observed[:,1]
    return torch.where(impossible,torch.full_like(result,torch.inf),result)


class FactorModel(nn.Module):
    def __init__(self,input_dim,geometry):
        super().__init__();self.geometry=geometry
        self.head=GeometryHead(input_dim) if geometry else nn.Sequential(
            nn.Linear(input_dim+3,136),nn.SiLU(),nn.Linear(136,136),nn.SiLU(),nn.Linear(136,1))

    def forward(self,context,physical,query,present):
        q=query[:,None].expand(-1,context.shape[1],-1)
        values=self.head(context[present],physical[present],q[present]) if self.geometry else self.head(
            torch.cat([context[present],q[present]],-1)).squeeze(-1)
        output=context.new_zeros(present.shape)
        return output.masked_scatter(present,values)


class SetModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(input_dim+3,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.rho=nn.Sequential(nn.Linear(64*3+4,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,context,query,keep):
        q=query[:,None].expand(-1,context.shape[1],-1)
        z=self.phi(torch.cat([context,q],-1));count=keep.sum(-1,keepdim=True)
        total=(z*keep[...,None]).sum(1);mean=total/count.clamp(min=1)
        maximum=z.masked_fill(~keep[...,None],-torch.inf).max(1).values
        maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        result=self.rho(torch.cat([total,mean,maximum,query,count.float()],-1)).squeeze(-1)
        return torch.where(count[:,0]>0,result,torch.full_like(result,-1000.))
