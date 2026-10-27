"""Train shared spectral cache scales on a real frozen VLA.

Exploratory low-rank compression baseline versus intervention-response calibration.
Teacher fidelity only; no claims of correct actions, safety, speedup or novelty.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from cavla.cache_transport import distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--steps', type=int, default=36)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--methods', nargs='+', default=['svd','endpoint','response','reweighted_endpoint'])
    p.add_argument('--evaluation-seeds', nargs='+', type=int, default=list(range(9201,9209)))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'data_manifest_sha256': hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        'training': 'Three shared scale tensors, same rank/steps/data/order; fixed endpoint selected',
        'scope': 'Six-scene exploratory frozen-policy fidelity pilot; not a validated ICML candidate',
        'calibration_seeds': [301, 302], 'evaluation_seeds': args.evaluation_seeds,
        'learning_rate': .02, 'contrast_weight': .25, 'energy_floor': .05}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    source = Path('/mnt/datasets/Explainability/alpamayo/src')
    checkpoint = '/home/perceptron-lab/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/snapshots/d72c6a2757b10cacb69b688ca66ccfad3dafbf37'
    sys.path.insert(0, str(source))
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    torch.set_num_threads(2)
    model = AlpamayoR1.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    data = []
    for path in sorted(args.data.glob('*_contexts.pt')):
        d = torch.load(path, map_location='cuda', weights_only=True)
        d['factors'] = {}
        with torch.no_grad():
            for name in ['original', 'edited']:
                factors = []
                for layer in d[name]['cache']:
                    fields = []
                    for field in ['keys', 'values']:
                        x = layer[field].float()
                        mean = x.mean(-2, keepdim=True)
                        centered = x-mean
                        _, eig = torch.linalg.eigh(centered.transpose(-1,-2) @ centered)
                        basis = eig[..., -args.rank:]
                        fields.append((mean.to(torch.bfloat16), (centered@basis).to(torch.bfloat16), basis.to(torch.bfloat16)))
                    factors.append(fields)
                d['factors'][name] = factors
        data.append(d)
        print(json.dumps({'factored': d['pair']}), flush=True)
    assert sum(d['pair']['split']=='train' for d in data)==3
    assert sum(d['pair']['split']=='test' for d in data)==3

    def cache_for(d, name, scales=None):
        tensors = []
        for i, layer in enumerate(d[name]['cache']):
            fields = []
            for j, field in enumerate(['keys','values']):
                if scales is None:
                    value = layer[field].clone()
                else:
                    mean, left, basis = d['factors'][name][i][j]
                    weighted = left * scales[i,j][None,:,None,:].to(left.dtype)
                    value = mean + weighted @ basis.transpose(-1,-2)
                fields.append(value)
            tensors.append(tuple(fields))
        return DynamicCache(tensors)

    def field(d, name, x, t, scales=None):
        ctx = d[name]
        cache = cache_for(d, name, scales)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embeds = model.action_in_proj(x,t)
            n = model.action_space.get_action_space_dims()[0]
            if embeds.ndim==2:
                embeds = embeds.view(x.shape[0], n, -1)
            output = model.expert(inputs_embeds=embeds, position_ids=ctx['position_ids'],
                past_key_values=cache, attention_mask=ctx['attention_mask'], use_cache=True,
                **ctx['forward_kwargs'])
            pred = model.action_out_proj(output.last_hidden_state[:,-n:]).view(-1,*model.action_space.get_action_space_dims())
        return pred.float()

    def rollout(d, name, seed, scales=None, return_action=False):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            action = model.diffusion.sample(batch_size=1,
                step_fn=lambda x,t: field(d,name,x,t,scales), device='cuda', return_all_steps=False)
            if return_action:
                return action
            xyz,_ = model.action_space.action_to_traj(action,d['history_xyz'],d['history_rot'])
        return xyz.float().cpu().numpy().reshape(-1,3)

    replay_checks=[]
    for d in data:
        path=args.data/f"{d['pair']['scene']}_trajectories.npz"
        with np.load(path) as reference:
            for name in ['original','edited']:
                error=distance(rollout(d,name,9101),reference[f'{name}_9101'])
                replay_checks.append({'scene':d['pair']['scene'],'condition':name,'error_m':error})
                if error>1e-4:raise RuntimeError(f'Serialized-cache replay failed: {replay_checks[-1]}')
    (args.out/'replay_checks.json').write_text(json.dumps(replay_checks,indent=2))
    calibration = []
    for i,d in enumerate(data):
        if d['pair']['split']!='train' or args.steps==0:
            continue
        for seed in [301,302]:
            clean = rollout(d,'original',seed,return_action=True)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            noise = torch.randn_like(clean)
            for time in [.1,.5,.9]:
                t = torch.full((1,1,1),time,device='cuda')
                x = (1-time)*noise + time*clean
                with torch.no_grad():
                    y0=field(d,'original',x,t); y1=field(d,'edited',x,t)
                calibration.append((i,x.detach(),t,y0.detach(),y1.detach()))
    print(json.dumps({'calibration_points':len(calibration)}),flush=True)
    n_layers=len(data[0]['original']['cache'])
    n_heads=data[0]['original']['cache'][0]['keys'].shape[1]
    fixed=torch.ones(n_layers,2,n_heads,args.rank,device='cuda')
    fitted={'svd':fixed}
    order=list(range(len(calibration))); random.Random(args.seed).shuffle(order)
    logs=[]
    for method in [m for m in args.methods if m!='svd']:
        scales=torch.nn.Parameter(fixed.clone())
        optimizer=torch.optim.Adam([scales],lr=.02)
        for step in range(args.steps):
            i,x,t,y0,y1=calibration[order[step%len(order)]]
            optimizer.zero_grad(set_to_none=True)
            s0=field(data[i],'original',x,t,scales)
            s1=field(data[i],'edited',x,t,scales)
            endpoint=((s0-y0).square().mean()+(s1-y1).square().mean())/2
            energy=(y0.square().mean()+y1.square().mean())/2+.05
            gap_energy=(y1-y0).square().mean()+.05
            gap_error=((s1-s0)-(y1-y0)).square().mean()
            if method=='reweighted_endpoint':
                loss=endpoint/energy+.25*endpoint/gap_energy
            else:
                loss=endpoint/energy+(.25*gap_error/gap_energy if method=='response' else 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([scales],1.)
            optimizer.step()
            with torch.no_grad():
                scales.clamp_(0,2)
            logs.append({'method':method,'step':step,'scene':data[i]['pair']['scene'],
                'loss':float(loss.detach()),'endpoint':float(endpoint.detach()),'gap':float(gap_error.detach())})
            if step%6==0: print(json.dumps(logs[-1]),flush=True)
        fitted[method]=scales.detach().clone()
        torch.save({'scales':fitted[method].cpu(),'rank':args.rank},args.out/f'{method}.pt')
        del scales,optimizer
        gc.collect()
    (args.out/'training.json').write_text(json.dumps(logs,indent=2))
    rows=[]
    for d in data:
        if d['pair']['split']!='test':continue
        saved={}
        for seed in args.evaluation_seeds:
            teacher={name:rollout(d,name,seed) for name in ['original','edited']}
            for method,scales in fitted.items():
                student={name:rollout(d,name,seed,scales) for name in ['original','edited']}
                effect_error=distance(student['edited']-student['original'],teacher['edited']-teacher['original'])
                row={'scene':d['pair']['scene'],'seed':seed,'method':method,
                    'factual_error_m':distance(student['original'],teacher['original']),
                    'edited_error_m':distance(student['edited'],teacher['edited']),
                    'effect_error_m':effect_error,
                    'teacher_effect_m':distance(teacher['original'],teacher['edited'])}
                rows.append(row)
                for name,x in student.items():saved[f'{method}_{name}_{seed}']=x
            for name,x in teacher.items():saved[f'teacher_{name}_{seed}']=x
        np.savez_compressed(args.out/f"{d['pair']['scene']}_predictions.npz",**saved)
        (args.out/'evaluation.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({'evaluated':d['pair']['scene']}),flush=True)
    summary={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in
        ['factual_error_m','edited_error_m','effect_error_m']} for m in fitted}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
