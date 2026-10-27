"""Compatibility gate for a deliberately strong fixed-route earliest-time model.

Descriptive policy samples under imperfect image edits, not physical worlds or
correct action labels. Failing this gate rejects that policy surrogate, not the
underlying feasibility constraints. Original-future selection is diagnostic only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def first_crossing(progress, distance):
    hits = np.flatnonzero(np.asarray(progress) >= distance)
    return .1 * (int(hits[0]) + 1) if len(hits) else float('inf')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, default=Path('artifacts/real_audit_20261020/pairs.csv'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes(); (args.out/'source_snapshot.py').write_bytes(source)
    rows = list(csv.DictReader(args.pairs.open())); cache = {}; records = []
    for row in rows:
        if row['label'] not in ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'person']:
            continue
        path = Path(row['trajectory_path']); original_path = path.parent/'original_traj.npz'
        if str(original_path) not in cache:
            with np.load(original_path) as z: original = z['pred_xyz'][0, :, :2].astype(float)
            assert original.shape == (64, 2)
            length = np.linalg.norm(original[-1]); axis = original[-1]/max(length, 1e-12)
            progress = original@axis
            lateral = np.abs(original@np.array([-axis[1], axis[0]]))
            eligible = length >= 5 and lateral.max() <= 1 and np.diff(progress).min() >= -.25
            cache[str(original_path)] = (axis, progress, bool(eligible))
        axis, original_progress, eligible = cache[str(original_path)]
        if not eligible: continue
        with np.load(path) as z: edited = z['pred_xyz'][0, :, :2].astype(float)
        assert edited.shape == (64, 2)
        edited_progress = edited@axis; times = []
        for distance in [2., 5., 10., 20.]:
            t0 = first_crossing(original_progress, distance)
            if t0 > 5.8: continue  # leave at least .6s to observe a violation
            t1 = first_crossing(edited_progress, distance)
            times.append({'distance': distance, 'original_time': t0,
                'edited_time': None if not np.isfinite(t1) else t1,
                'delay_over_half_second': bool(t1 > t0 + .5 + 1e-8)})
        records.append({'scene': row['scene'], 'stem': row['stem'], 'label': row['label'],
            'any_delay_over_half_second': any(r['delay_over_half_second'] for r in times),
            'queries': times})
    grouped = {}
    for row in records: grouped.setdefault(row['scene'], []).append(row['any_delay_over_half_second'])
    result = {'hypothesis': 'Sampled policy uses a fixed route, selects earliest feasible event times, and an image edit only deletes constraints',
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'pairs_sha256': hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        'originals': len(cache), 'eligible_originals': sum(v[2] for v in cache.values()),
        'eligible_pairs': len(records), 'pairs_with_delay': sum(r['any_delay_over_half_second'] for r in records),
        'scenes_with_any_delay': sum(any(v) for v in grouped.values()),
        'scene_mean_fraction_pairs_with_delay': float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None,
        'limitations': ['Existing single-seed archived trajectories; no paired independent distribution estimate',
            'Image deletion may change pixels outside the object and is not a verified constraint deletion',
            'Selected action may change route, utility or risk preference; physical feasible sets are not identified',
            'Straight/moving original-future selection is diagnostic, never a prospective cohort',
            'This does not refute feasibility monotonicity or establish that any delayed action is unsafe']}
    (args.out/'summary.json').write_text(json.dumps(result, indent=2))
    (args.out/'records.json').write_text(json.dumps(records, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
