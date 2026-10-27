"""Fully labelled utility diagnostic, not label-blind active acquisition."""
import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from cavla.active_pilot import flatten, tensorize, save_json
from cavla.controlled import world_effect
from cavla.pilot import Policy, predict, dataset_digest
from cavla.relations import build, states, quartets

PROXIES = ('action_error', 'effect_error', 'relation_violation', 'gradient_alignment')


def balanced(seed, n_each):
    pieces = [quartets(states(n_each, seed+2*i), seed+2*i+1, family)
              for i, family in enumerate(('approach', 'background', 'billboard', 'lane_swap'))]
    return {k: np.concatenate([d[k] for d in pieces]) for k in pieces[0]}


def vector_gradient(model, loss):
    # Unused auxiliary head is excluded consistently in every gradient vector.
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return torch.cat([g.reshape(-1) for g in gradients if g is not None]).detach()


def scores(model, candidate, calibration, factual, device):
    model.eval()
    prediction = predict(model, tensorize(candidate, device))
    effect, truth = world_effect(prediction), world_effect(candidate['targets'])
    relation = candidate['relation']
    zero = relation == 0
    penalties = np.where(zero, effect**2, np.maximum(0., .01-effect*relation)**2)
    # Mean per present category, matching relation_loss; empty categories = 0.
    violation = np.array([penalties[i][zero[i]].mean() +
                          (penalties[i][~zero[i]].mean() if (~zero[i]).any() else 0.)
                          for i in range(len(candidate['images']))])
    cal_prediction, _ = model(calibration['images'], calibration['commands'])
    grad_cal = vector_gradient(model, F.mse_loss(cal_prediction, calibration['targets']))
    factual_prediction, _ = model(factual['images'], factual['commands'])
    grad_factual = vector_gradient(model, F.mse_loss(factual_prediction, factual['targets']))
    full = tensorize(candidate, device)
    align = []
    for i in range(len(prediction)):
        p, _ = model(full['images'][i:i+1], full['commands'][i:i+1])
        grad = vector_gradient(model, F.mse_loss(p, full['targets'][i:i+1]))
        align.append(float(torch.dot(.5*(grad+grad_factual), grad_cal)))
    return dict(action_error=((prediction[:, [1, 3]]-candidate['targets'][:, [1, 3]])**2).mean(axis=(1, 2)),
                effect_error=((effect-truth)**2).mean(axis=(1, 2)),
                relation_violation=violation, gradient_alignment=np.array(align))


def microadapt(model, candidate, factual, seed, steps=40):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    model.train()
    for _ in range(steps):
        ix = torch.randint(len(factual['images']), (32,), generator=rng).to(device)
        optimizer.zero_grad(set_to_none=True)
        p, _ = model(candidate['images'], candidate['commands'])
        replay, _ = model(factual['images'][ix], factual['commands'][ix])
        loss = .5*(F.mse_loss(p, candidate['targets'])+F.mse_loss(replay, factual['targets'][ix]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()


def run(args):
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Fresh output required')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    protocol = Path('docs/PROTOCOLO_UTILIDADE_V1.md')
    shutil.copyfile(protocol, args.out/protocol.name)
    snapshot = args.out/'source_snapshot'; snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'): shutil.copyfile(source, snapshot/source.name)
    save_json(args.out/'manifest.json', dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
              source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},
              annotation_access='ALL 48 candidates labelled: 96 factual + 96 edited action vectors per seed; 384 calibration action vectors; 448 original training action vectors',
              limitations=['Same synthetic CNN task as relation pilot', 'Fully labelled diagnosis, no annotation savings',
                           'Gain measured on calibration, no held-out selector generalization claim']))
    rows = []
    for seed in args.seeds:
        origin = 50_000_000+seed*100
        train, _, _, _ = build(seed, 12)
        candidate, cal = balanced(origin, 12), balanced(origin+20, 24)
        if set(candidate['ids']) & set(cal['ids']): raise ValueError('Calibration leakage')
        known, calibration = tensorize(flatten(train, views=[0, 2]), device), tensorize(cal, device)
        folder = args.out/f'seed{seed}'; folder.mkdir()
        checkpoint = args.checkpoints/f'seed{seed}'/'pretrained.pt'
        initial = torch.load(checkpoint, map_location=device, weights_only=True)
        model = Policy().to(device); model.load_state_dict(initial)
        baseline = float(np.mean((predict(model, calibration)-cal['targets'])**2))
        proxies = scores(model, candidate, calibration, known, device)
        full = tensorize(candidate, device)
        gains = []
        for i in range(len(candidate['images'])):
            model = Policy().to(device); model.load_state_dict(initial)
            microadapt(model, {k: v[i:i+1] for k, v in full.items()}, known, seed+90000, args.steps)
            after = float(np.mean((predict(model, calibration)-cal['targets'])**2))
            gains.append(baseline-after)
        gains = np.array(gains)
        row = dict(seed=seed, baseline_mse=baseline, mean_gain=float(gains.mean()),
                   positive_fraction=float((gains > 0).mean()),
                   proxies={k: dict(spearman=float(spearmanr(v, gains).statistic),
                                    top8_mean_gain=float(gains[np.argsort(v)[-8:]].mean()))
                            for k, v in proxies.items()})
        rows.append(row)
        save_json(folder/'inputs.json', dict(candidate_hash=dataset_digest(candidate), calibration_hash=dataset_digest(cal),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        np.savez_compressed(folder/'candidate_utility.npz', **proxies, gain=gains, families=candidate['family'], ids=candidate['ids'])
        save_json(args.out/'runs.json', rows)
        print(row, flush=True)
    summary = {'mean_spearman': {k: float(np.mean([r['proxies'][k]['spearman'] for r in rows])) for k in PROXIES}}
    summary['gates'] = {}
    for k in ('effect_error', 'relation_violation'):
        delta = [r['proxies'][k]['spearman']-r['proxies']['action_error']['spearman'] for r in rows]
        positive = sum(r['proxies'][k]['top8_mean_gain'] > 0 for r in rows)
        summary['gates'][k] = dict(mean_correlation_improvement=float(np.mean(delta)),
                                  wins=sum(d > 0 for d in delta), positive_top8=positive,
                                  pass_gate=float(np.mean(delta)) >= .1 and sum(d > 0 for d in delta) >= 4 and positive >= 4)
    save_json(args.out/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=Path('artifacts/relations_v1_20261020'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[616, 717, 818, 919, 1020])
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
