"""Falsify joint trust learning against generic filtering; not a new VLA method."""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cavla.active_pilot import flatten, tensorize, train_model, response_and_features, save_json
from cavla.controlled import world_effect
from cavla.pilot import Policy, predict, dataset_digest
from cavla.relations import build, weak_view, metrics

METHODS = ('factual', 'naive', 'audited_only', 'hard_filter', 'soft_filter',
           'anchor_soft', 'robust_fixed', 'joint_mixture', 'oracle_filter')
REGIMES = ('clean', 'random25', 'structured25')


def corrupt(pool, regime, seed):
    """Reference-side generator; return observed data and sealed audit truth."""
    observed = {k: v.copy() for k, v in pool.items()}
    if regime == 'random25': candidates = np.arange(len(pool['images']))
    elif regime in ('structured25', 'structured_matched'): candidates = np.where(pool['family'] == 'approach')[0]
    elif regime == 'clean': candidates = np.array([], dtype=int)
    else: raise ValueError(regime)
    selected = (np.random.default_rng(seed+30000).choice(candidates, 24, replace=False)
                if regime != 'clean' else np.array([], dtype=int))
    # Only the independent generator uses states/true relations to assign noise.
    # The blocked command is recognizable from its lower factual forward target.
    blocked = np.argmin(pool['targets'][:, [0, 2], 1], axis=1)
    observed['relation'][selected, blocked[selected], 1] = 0 if regime == 'structured_matched' else 1
    good = (observed['relation'] == pool['relation']).all(axis=(1, 2))
    audit_indices = np.random.default_rng(seed+40000).choice(len(good), 32, replace=False)
    return observed, dict(indices=audit_indices, good=good[audit_indices].astype(np.float32)), good


def features(model, pool, device):
    response, z, _ = response_and_features(model, pool, device)
    rgb = np.abs(pool['images'][:, 1]-pool['images'][:, 0]).mean(axis=(2, 3))
    return np.c_[z, response, pool['relation'].reshape(len(z), -1), rgb].astype(np.float32)


def trust_model(x, audit, seed, device):
    torch.manual_seed(seed+50000)
    head = nn.Sequential(nn.Linear(x.shape[1], 32), nn.SiLU(), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
    ix = torch.as_tensor(audit['indices'], device=device)
    labels = torch.as_tensor(audit['good'], device=device)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(x[ix]).squeeze(-1), labels)
        loss.backward(); optimizer.step()
    return head


def per_pair_loss(effect, relation):
    zero = relation == 0
    direction = ~zero
    null_term = (effect.square()*zero).sum((1, 2))/zero.sum((1, 2)).clamp_min(1)
    sign_term = (F.relu(.01-effect*relation).square()*direction).sum((1, 2))/direction.sum((1, 2)).clamp_min(1)
    return null_term+sign_term


def mixture_loss(losses, probability, tau=.0001):
    p = probability.clamp(1e-6, 1-1e-6)
    mixed = -tau*torch.logaddexp(p.log()-losses/tau, torch.log1p(-p))
    # Audited certainty is exact: p=1 cannot discard a hard approved constraint.
    return torch.where(probability >= 1, losses, torch.where(probability <= 0, losses*0, mixed))


def train_candidate(model, head, known, weak, x, audit, validation, method, seed, steps, oracle_good=None):
    if method not in METHODS: raise ValueError(method)
    if method == 'oracle_filter' and oracle_good is None: raise ValueError('Oracle comparator needs explicit truth')
    device = next(model.parameters()).device
    audit_ix = torch.as_tensor(audit['indices'], device=device)
    audit_labels = torch.as_tensor(audit['good'], device=device)
    known_quality = torch.full((len(x),), -1., device=device)
    known_quality[audit_ix] = audit_labels
    optimizer = torch.optim.AdamW(list(model.parameters())+(list(head.parameters()) if method == 'joint_mixture' else []),
                                  lr=1e-3, weight_decay=1e-4)
    if method != 'joint_mixture':
        for param in head.parameters(): param.requires_grad_(False)
    rng = torch.Generator().manual_seed(seed+60000)
    best, best_step, saved, saved_head = float('inf'), 0, None, None
    curve = []; started = time.perf_counter()
    for step in range(1, steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        ix = torch.randint(len(known['images']), (64 if method == 'factual' else 32,), generator=rng).to(device)
        pred, _ = model(known['images'][ix], known['commands'][ix])
        bc = F.mse_loss(pred, known['targets'][ix]); loss = bc
        if method != 'factual':
            pi = torch.randint(len(x), (32,), generator=rng).to(device)
            p, _ = model(weak['images'][pi], weak['commands'][pi])
            effect = p[:, [1, 3]]-weak['anchor'][pi] if method == 'anchor_soft' else world_effect(p)
            constraints = per_pair_loss(effect, weak['relation'][pi])
            probability = torch.sigmoid(head(x[pi]).squeeze(-1))
            probability = torch.where(known_quality[pi] >= 0, known_quality[pi], probability)
            if method == 'robust_fixed':
                probability = torch.where(known_quality[pi] >= 0, known_quality[pi], audit_labels.mean())
            if method in ('joint_mixture', 'robust_fixed'):
                extra = mixture_loss(constraints, probability).mean()
            else:
                if method == 'naive': weights = torch.ones_like(probability)
                elif method == 'audited_only': weights = (known_quality[pi] == 1).float()
                elif method == 'hard_filter': weights = (probability >= .5).float()
                elif method == 'oracle_filter': weights = oracle_good[pi]
                else: weights = probability
                extra = (constraints*weights).sum()/weights.sum().clamp_min(1)
            loss = bc+.5*extra
            if method == 'joint_mixture':
                loss = loss+.001*F.binary_cross_entropy_with_logits(head(x[audit_ix]).squeeze(-1), audit_labels)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if step % 100 == 0 or step == steps:
            value = float(np.mean((predict(model, validation)-validation['targets'].cpu().numpy())**2))
            curve.append(dict(step=step, factual_mse=bc.item(), loss=loss.item(), validation_mse=value))
            if value < best:
                best, best_step = value, step
                saved, saved_head = copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())
    model.load_state_dict(saved); head.load_state_dict(saved_head)
    if device.type == 'cuda': torch.cuda.synchronize()
    return dict(best_step=best_step, validation_mse=best, seconds=time.perf_counter()-started, curve=curve)


def summarize(rows):
    regimes = sorted(set(r['regime'] for r in rows))
    primary_regime = 'structured_matched' if 'structured_matched' in regimes else 'structured25'
    means = {regime: {method: float(np.mean([r['metrics']['primary'] for r in rows
                     if r['regime'] == regime and r['method'] == method])) for method in METHODS} for regime in regimes}
    comparisons = {}
    candidate = {r['seed']: r for r in rows if r['regime'] == primary_regime and r['method'] == 'joint_mixture'}
    for method in ('soft_filter', 'anchor_soft', 'robust_fixed'):
        control = {r['seed']: r for r in rows if r['regime'] == primary_regime and r['method'] == method}
        comparisons[method] = dict(relative_reduction=1-means[primary_regime]['joint_mixture']/means[primary_regime][method],
             wins=sum(candidate[s]['metrics']['primary'] < control[s]['metrics']['primary'] for s in candidate))
    retention = np.mean([r['metrics']['id']['action_mse'] for r in candidate.values()]) <= 1.1*np.mean(
        [r['metrics']['id']['action_mse'] for r in rows if r['regime'] == primary_regime and r['method'] == 'soft_filter'])
    return dict(means=means, candidate_gate=dict(comparisons=comparisons, retention_pass=bool(retention),
                pass_gate=bool(retention and all(v['relative_reduction'] >= .1 and v['wins'] >= 4 for v in comparisons.values()))))


def run(args):
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Fresh output required')
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    protocol = Path('docs/PROTOCOLO_SUPERVISAO_INCERTA_V1.md')
    shutil.copyfile(protocol, args.out/protocol.name)
    snapshot = args.out/'source_snapshot'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path, snapshot/path.name)
    save_json(args.out/'manifest.json', dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
         source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},
         scope='Controlled CNN: incorrect relation supervision, not semantically invalid tasks or VLA',
         torch=torch.__version__, audit_cost='32 independent binary quality labels per regime and seed; no corrected actions or relations'))
    rows = []
    for seed in args.seeds:
        train, pool, validation, tests = build(seed)
        folder = args.out/f'seed{seed}'; folder.mkdir()
        save_json(folder/'dataset_hashes.json', {k: dataset_digest(v) for k,v in dict(train=train,pool=pool,validation=validation,**tests).items()})
        known, val = tensorize(flatten(train, views=[0, 2]), device), tensorize(flatten(validation, views=[0, 2]), device)
        torch.manual_seed(seed)
        model = Policy().to(device)
        if args.initial_from is not None:
            checkpoint = args.initial_from/f'seed{seed}'/'pretrained.pt'
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
            info = dict(reused_checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        else:
            info = train_model(model, known, val, args.pretrain_steps, seed+70000)
        initial = copy.deepcopy(model.state_dict()); torch.save(initial, folder/'pretrained.pt')
        save_json(folder/'pretrained.json', info)
        for regime in args.regimes:
            observed, audit, good = corrupt(pool, regime, seed)
            weak = weak_view(observed, device)
            x = features(model, observed, device)
            mean, std = x.mean(0), x.std(0).clip(.001)
            x_t = torch.as_tensor((x-mean)/std, device=device)
            trust = trust_model(x_t, audit, seed, device)
            trust_state = copy.deepcopy(trust.state_dict())
            torch.save(trust_state, folder/f'{regime}_initial_trust.pt')
            np.savez_compressed(folder/f'{regime}_audits_and_reference.npz', audit_indices=audit['indices'],
                    audit_good=audit['good'], reference_good=good, observed_relation=observed['relation'], features=x)
            for method in METHODS:
                model = Policy().to(device); model.load_state_dict(initial)
                head = nn.Sequential(nn.Linear(x.shape[1], 32), nn.SiLU(), nn.Linear(32, 1)).to(device)
                head.load_state_dict(trust_state)
                # Only the explicitly privileged comparator receives hidden quality.
                quality = torch.as_tensor(good.astype(np.float32), device=device) if method == 'oracle_filter' else None
                info = train_candidate(model, head, known, weak, x_t, audit, val, method, seed, args.adapt_steps, quality)
                values, arrays = metrics(model, tests, device)
                with torch.inference_mode(): probability = torch.sigmoid(head(x_t).squeeze(-1)).cpu().numpy()
                hidden = np.ones(len(good), dtype=bool); hidden[audit['indices']] = False
                quality_metrics = dict(hidden_brier=float(np.mean((probability[hidden]-good[hidden])**2)),
                    hidden_good_mean=float(probability[hidden & good].mean()) if (hidden & good).any() else None,
                    hidden_bad_mean=float(probability[hidden & ~good].mean()) if (hidden & ~good).any() else None)
                rows.append(dict(seed=seed, regime=regime, method=method, metrics=values, training=info, trust=quality_metrics))
                save_json(args.out/'runs.json', rows)
                np.savez_compressed(folder/f'{regime}_{method}_test.npz', **arrays, trust_probability=probability)
                torch.save(dict(policy=model.state_dict(), trust=head.state_dict()), folder/f'{regime}_{method}.pt')
                print(dict(seed=seed,regime=regime,method=method,primary=values['primary']),flush=True)
            # Feature extraction for subsequent regimes must use the factual checkpoint.
            model = Policy().to(device); model.load_state_dict(initial)
    save_json(args.out/'summary.json', summarize(rows))
    print(summarize(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[1636,1747,1858,1969,2080])
    parser.add_argument('--pretrain-steps',type=int,default=2000)
    parser.add_argument('--adapt-steps',type=int,default=1200)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--regimes',nargs='+',choices=(*REGIMES,'structured_matched'),default=list(REGIMES))
    parser.add_argument('--initial-from',type=Path)
    run(parser.parse_args())
