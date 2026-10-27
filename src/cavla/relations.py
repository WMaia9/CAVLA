"""Preregistered weak-relation pilot; weak batches never contain edited actions."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from cavla.controlled import State, oracle, render, world_effect
from cavla.pilot import Policy, predict, dataset_digest
from cavla.active_pilot import flatten, tensorize, train_model, save_json

METHODS = ('factual_only', 'null_copy', 'weak_null', 'weak_relation',
           'null_copy_relation', 'all_invariance', 'full_cf16', 'full_cf96')


def states(n, seed):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n):
        lane = int(rng.choice([-1, 1]))
        result.append(State(lane, float(rng.uniform(.6, .9)),
                            float((.08 if lane == -1 else .30)+rng.uniform(-.025, .025)),
                            float(rng.uniform(.1, .9))))
    return result


def quartets(base_states, seed, mode='id', ids=None):
    rng = np.random.default_rng(seed)
    images, targets, families = [], [], []
    for i, base in enumerate(base_states):
        family = ('approach', 'background', 'billboard')[i % 3] if mode == 'id' else mode
        if family in ('approach', 'severity'):
            lo, hi = (.24, .4) if family == 'severity' else (.04, .18)
            changed = replace(base, distance=base.distance-float(rng.uniform(lo, hi)))
        elif family == 'background': changed = replace(base, background=.38-base.background)
        elif family == 'billboard': changed = replace(base, billboard=1-base.billboard)
        elif family == 'lane_swap': changed = replace(base, blocker_lane=-base.blocker_lane)
        elif family == 'composition':
            changed = replace(base, blocker_lane=-base.blocker_lane, background=.38-base.background)
        else: raise ValueError(family)
        images.append(np.stack([render(base), render(changed), render(base), render(changed)]))
        targets.append(np.stack([oracle(base, 0), oracle(changed, 0), oracle(base, 1), oracle(changed, 1)]))
        families.append(family)
    targets = np.stack(targets)
    effects = world_effect(targets)
    nonzero = np.abs(effects) > 1e-6
    if np.any(np.abs(effects[nonzero]) < .01): raise ValueError('Relation below resolution')
    return dict(images=np.stack(images), commands=np.tile([0, 0, 1, 1], (len(targets), 1)),
                targets=targets, relation=np.where(nonzero, np.sign(effects), 0).astype(np.int8),
                null=~nonzero.any(axis=(1, 2)), family=np.array(families),
                ids=np.array(ids if ids is not None else [f'{seed}:{i}' for i in range(len(targets))]))


def build(seed, test_size=256):
    origin = 30_000_000+seed*100
    base = states(224, origin)
    train = quartets(base, origin+1, ids=[f'{origin}:{i}' for i in range(224)])
    chosen = np.random.default_rng(origin+2).choice(224, 96, replace=False)
    pool = quartets([base[i] for i in chosen], origin+3, ids=train['ids'][chosen])
    validation = quartets(states(96, origin+4), origin+5)
    tests = {name: quartets(states(test_size, origin+20+2*i), origin+21+2*i, name)
             for i, name in enumerate(('id', 'severity', 'lane_swap', 'composition'))}
    partitions = [set(d['ids']) for d in [train, validation, *tests.values()]]
    if any(a & b for i, a in enumerate(partitions) for b in partitions[i+1:]):
        raise ValueError('Partition overlap')
    return train, pool, validation, tests


def weak_view(pool, device):
    # Whitelist fields: changed-world labels cannot accidentally enter training.
    return {'images': torch.from_numpy(pool['images']).to(device),
            'commands': torch.from_numpy(pool['commands']).to(device),
            'relation': torch.from_numpy(pool['relation']).to(device),
            'anchor': torch.from_numpy(pool['targets'][:, [0, 2]].copy()).to(device)}


def relation_loss(effect, relation, include_zero=True, include_direction=True):
    zero = relation == 0
    loss = effect.sum()*0
    if include_zero and zero.any(): loss = loss+effect[zero].square().mean()
    nonzero = ~zero
    if include_direction and nonzero.any():
        loss = loss+F.relu(.01-effect[nonzero]*relation[nonzero]).square().mean()
    return loss


def adapt(model, known, weak, validation, method, steps, seed, full=None):
    if method not in (*METHODS, 'anchor_relation') or steps < 1:
        raise ValueError('Known method and positive adaptation steps required')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    null_indices = torch.where((weak['relation'] == 0).all(dim=(1, 2)))[0]
    best, best_step, saved, curve = float('inf'), 0, None, []
    if device.type == 'cuda': torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        ix = torch.randint(len(known['images']), (64 if method == 'factual_only' else 32,), generator=rng).to(device)
        prediction, _ = model(known['images'][ix], known['commands'][ix])
        loss = F.mse_loss(prediction, known['targets'][ix])
        if method.startswith('full_cf'):
            if full is None: raise ValueError('Explicit exact supervision required')
            ix = torch.randint(len(full['images']), (32,), generator=rng).to(device)
            p, _ = model(full['images'][ix], full['commands'][ix])
            loss = .5*(loss+F.mse_loss(p, full['targets'][ix]))
        elif method != 'factual_only':
            ix = torch.randint(len(weak['images']), (32,), generator=rng).to(device)
            if method == 'null_copy':
                ix = null_indices[torch.randint(len(null_indices), (32,), generator=rng).to(device)]
            p, _ = model(weak['images'][ix], weak['commands'][ix])
            effect, relation = world_effect(p), weak['relation'][ix]
            extra = loss*0
            if method in ('null_copy', 'null_copy_relation'):
                mask = (relation == 0).all(dim=(1, 2))
                if mask.any(): extra = extra+F.mse_loss(p[mask][:, [1, 3]], weak['anchor'][ix][mask])
            if method in ('weak_null', 'weak_relation', 'null_copy_relation'):
                extra = extra+relation_loss(effect, relation,
                                           include_zero=method != 'null_copy_relation',
                                           include_direction=method != 'weak_null')
            if method == 'anchor_relation':
                extra = extra+relation_loss(p[:, [1, 3]]-weak['anchor'][ix], relation)
            if method == 'all_invariance': extra = effect.square().mean()
            loss = loss+.5*extra
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if step % 100 == 0 or step == steps:
            value = float(np.mean((predict(model, validation)-validation['targets'].cpu().numpy())**2))
            curve.append({'step': step, 'loss': loss.item(), 'validation_mse': value})
            if value < best:
                best, best_step, saved = value, step, copy.deepcopy(model.state_dict())
    model.load_state_dict(saved)
    if device.type == 'cuda': torch.cuda.synchronize()
    return dict(seconds=time.perf_counter()-started, best_step=best_step, validation_mse=best, curve=curve)


def metrics(model, tests, device):
    result, arrays = {}, {}
    for name, data in tests.items():
        pred = predict(model, tensorize(data, device))
        error = ((pred-data['targets'])**2).mean(axis=(1, 2))
        effect, truth = world_effect(pred), world_effect(data['targets'])
        active = np.abs(truth) > 1e-6
        result[name] = dict(action_mse=float(error.mean()),
                            effect_rmse=float(np.sqrt(((effect-truth)**2).mean())),
                            direction_accuracy=float((effect[active]*truth[active] > 0).mean()),
                            zero_component_rms=float(np.sqrt(np.mean(effect[~active]**2))))
        arrays[name+'_prediction'] = pred
        arrays[name+'_targets'] = data['targets']
        arrays[name+'_scene_ids'] = data['ids']
        arrays[name+'_scene_mse'] = error
    result['primary'] = float(np.mean([result[k]['action_mse'] for k in ('severity', 'lane_swap', 'composition')]))
    return result, arrays


def summarize(rows):
    means = {m: {k: float(np.mean([r['metrics'][k] if k == 'primary' else r['metrics'][k]['action_mse']
                                 for r in rows if r['method'] == m]))
                 for k in ('primary', 'id', 'severity', 'lane_swap', 'composition')} for m in METHODS}
    gates = {}
    for method in ('weak_relation', 'null_copy_relation'):
        comparisons = {}
        for control in ('factual_only', 'null_copy'):
            candidate = {r['seed']: r['metrics']['primary'] for r in rows if r['method'] == method}
            baseline = {r['seed']: r['metrics']['primary'] for r in rows if r['method'] == control}
            comparisons[control] = dict(relative_reduction=1-means[method]['primary']/means[control]['primary'],
                                        wins=sum(candidate[s] < baseline[s] for s in candidate))
        retain = means[method]['id'] <= 1.1*means['null_copy']['id']
        gates[method] = dict(comparisons=comparisons, retention_pass=retain,
                             pass_gate=retain and all(v['relative_reduction'] >= .1 and v['wins'] >= 4 for v in comparisons.values()))
    return dict(means=means, gates=gates)


def run(args):
    if len(set(args.seeds)) != len(args.seeds) or min(args.pretrain_steps, args.adapt_steps, args.test_size) < 1:
        raise ValueError('Distinct seeds and positive sizes required')
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Output must be fresh')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    snapshot = args.out/'source_snapshot'; snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'): shutil.copyfile(source, snapshot/source.name)
    protocol = Path('docs/PROTOCOLO_RELACOES_V1.md')
    shutil.copyfile(protocol, args.out/protocol.name)
    save_json(args.out/'manifest.json', dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(), torch=torch.__version__,
              source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},
              gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              supervision='448 factual actions; 96x2x2 categorical relations; full_cf methods additionally 2 actions per edited scene',
              limitations=['Synthetic CNN with two command IDs, not VLA', 'Perfect simulated relation labels',
                           'Equal steps, unequal forward-pass cost; timings saved', 'No human annotation cost measured']))
    rows = []
    for seed in args.seeds:
        train, pool, validation, tests = build(seed, args.test_size)
        folder = args.out/f'seed{seed}'; folder.mkdir()
        save_json(folder/'dataset_hashes.json', {k: dataset_digest(d) for k, d in dict(train=train, pool=pool, validation=validation, **tests).items()})
        np.savez_compressed(folder/'relation_labels.npz', relation=pool['relation'], scene_ids=pool['ids'], families=pool['family'])
        known = tensorize(flatten(train, views=[0, 2]), device)
        val = tensorize(flatten(validation, views=[0, 2]), device)
        weak = weak_view(pool, device)
        torch.manual_seed(seed)
        model = Policy().to(device)
        info = train_model(model, known, val, args.pretrain_steps, seed+60000)
        initial = copy.deepcopy(model.state_dict())
        torch.save(initial, folder/'pretrained.pt'); save_json(folder/'pretrain.json', info)
        chosen = np.random.default_rng(seed+70000).permutation(96)
        save_json(folder/'full_cf16_indices.json', chosen[:16].tolist())
        for method in METHODS:
            model = Policy().to(device); model.load_state_dict(initial)
            full = None
            if method.startswith('full_cf'):
                n = int(method.removeprefix('full_cf'))
                full = tensorize(flatten(pool, scenes=chosen[:n]), device)
            info = adapt(model, known, weak, val, method, args.adapt_steps, seed+80000, full)
            values, arrays = metrics(model, tests, device)
            row = dict(seed=seed, method=method, metrics=values, training=info)
            rows.append(row); save_json(args.out/'runs.json', rows)
            np.savez_compressed(folder/f'{method}_test.npz', **arrays)
            torch.save(model.state_dict(), folder/f'{method}.pt')
            print(json.dumps(dict(seed=seed, method=method, primary=values['primary'], seconds=info['seconds'])), flush=True)
    result = summarize(rows)
    save_json(args.out/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[616, 717, 818, 919, 1020])
    parser.add_argument('--pretrain-steps', type=int, default=2000)
    parser.add_argument('--adapt-steps', type=int, default=1200)
    parser.add_argument('--test-size', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
