"""Post-screening controls, explicitly separate from the original gate."""
import copy
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch

from cavla.active_pilot import flatten, tensorize, save_json
from cavla.pilot import Policy
from cavla.relations import build, weak_view, adapt, metrics


def run():
    out = Path('artifacts/relations_controls_v1_20261020')
    if out.exists(): raise ValueError('Fresh output required')
    out.mkdir()
    protocol = Path('docs/ADENDO_RELACOES_CONTROLES.md')
    shutil.copyfile(protocol, out/protocol.name)
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    for source in Path(__file__).parent.glob('*.py'): shutil.copyfile(source, snapshot/source.name)
    save_json(out/'manifest.json', dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
              source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')},
              status='Exploratory follow-up controls after positive screening; no original-gate replacement'))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rows = []
    for seed in (616, 717, 818, 919, 1020, 1121, 1222, 1323, 1424, 1525):
        root = 'relations_v1_20261020' if seed <= 1020 else 'relations_confirmation_v1_20261020'
        checkpoint = Path('artifacts')/root/f'seed{seed}'/'pretrained.pt'
        initial = torch.load(checkpoint, map_location='cuda', weights_only=True)
        train, pool, validation, tests = build(seed)
        known = tensorize(flatten(train, views=[0, 2]), 'cuda')
        val = tensorize(flatten(validation, views=[0, 2]), 'cuda')
        weak = weak_view(pool, 'cuda')
        folder = out/f'seed{seed}'; folder.mkdir()
        for name, method, steps in [('factual_3x', 'factual_only', 3600), ('anchor_relation', 'anchor_relation', 1200)]:
            model = Policy().to('cuda'); model.load_state_dict(initial)
            training = adapt(model, known, weak, val, method, steps, seed+80000)
            values, arrays = metrics(model, tests, 'cuda')
            row = dict(seed=seed, method=name, training=training, metrics=values)
            rows.append(row); save_json(out/'runs.json', rows)
            np.savez_compressed(folder/f'{name}_test.npz', **arrays)
            torch.save(model.state_dict(), folder/f'{name}.pt')
            print(dict(seed=seed, method=name, primary=values['primary']), flush=True)


if __name__ == '__main__': run()
