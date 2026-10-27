"""Exploratory temporal-label audit; no trained model or causal safety claim.

Compare pinhole stationary geometry with privileged annotation controls. Past-CV
uses only annotations timestamped at/before the actual input camera timestamp.
Future-frozen is explicitly an oracle (first future box at t+0.1 seconds).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cavla.geometric_learning import query_curve
from cavla.nuscenes_geometry import iter_array, rotation, signed_point_box_distance


WINDOWS = [(0., 1.), (1., 2.), (2., 4.), (4., 6.4), (0., 6.4)]


def past_state(track, samples, timestamp, ego):
    past = sorted((samples[a['sample_token']]['timestamp'], a) for a in track
                  if samples[a['sample_token']]['timestamp'] <= timestamp)
    if len(past) < 2:
        return None
    (t0, a0), (t1, a1) = past[-2:]
    gap = (t1 - t0) * 1e-6
    age = (timestamp - t1) * 1e-6
    if not 0 < gap <= .75 or age > .75:
        return None
    inv = rotation(ego['rotation']).inv()
    velocity = inv.apply((np.array(a1['translation']) - a0['translation']) / gap)[:2]
    center = inv.apply(np.array(a1['translation']) - ego['translation'])[:2] + age * velocity
    yaw = (inv * rotation(a1['rotation'])).as_euler('xyz')[2]
    return center, velocity, yaw, np.array(a1['size']), {
        'latest_annotation_timestamp': t1, 'input_timestamp': timestamp,
        'velocity_interval_seconds': gap, 'latest_annotation_age_seconds': age}


def metrics(y, p):
    pos = int(y.sum()); neg = int((~y).sum())
    fn = int((y & ~p).sum()); fp = int((~y & p).sum())
    return {'positive': pos, 'negative': neg, 'false_negative': fn, 'false_positive': fp,
            'balanced_error': (fn / pos + fp / neg) / 2 if pos and neg else None,
            'disagreement': float((y != p).mean())}


def run(args):
    datasets = [Path('artifacts/partial_composition_data_v2_20261021'),
                Path('artifacts/composition_holdout_data_v1_20261021')]
    geoms = [Path('artifacts/nuscenes_geometry_v3_20261021'),
             Path('artifacts/composition_holdout_geometry_v1_20261021')]
    priors = json.loads(Path('artifacts/structured_geometry_log_validation_20261021/manifest.json').read_text())['class_priors']
    collections = []
    for data, geom in zip(datasets, geoms):
        with np.load(data / 'dataset.npz') as z:
            ids = np.unique(z['scene_object_ids'][z['scene_object_ids'] >= 0])
            queries = np.unique(z['queries'][:, :2], axis=0)
        objects = json.loads((data / 'objects.json').read_text())
        records = {(r['scene'], r['stem']): r for r in json.loads((geom / 'records.json').read_text())}
        collections.append([(int(i), objects[i], records[objects[i]['scene'], objects[i]['stem']]) for i in ids])
    wanted_instances = {r['instance_token'] for records in collections for _, _, r in records}
    wanted_scenes = {r['scene'] for records in collections for _, _, r in records}
    tables = args.nuscenes / 'v1.0-trainval'
    samples = {r['token']: r for r in json.loads((tables / 'sample.json').read_text())}
    root = Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen')
    ranked = {r['scene_name']: r for r in csv.DictReader((root / 'scenes_ranked.csv').open()) if r['scene_name'] in wanted_scenes}
    wanted_sd = {r['sd_token'] for r in ranked.values()}
    print('Scanning past-only annotation tracks and input timestamps', flush=True)
    tracks = {}
    for r in iter_array(tables / 'sample_annotation.json'):
        if r['instance_token'] in wanted_instances:
            tracks.setdefault(r['instance_token'], []).append(r)
    sd = {r['token']: r for r in iter_array(tables / 'sample_data.json') if r['token'] in wanted_sd}
    wanted_ego = {r['ego_pose_token'] for r in sd.values()}
    egos = {r['token']: r for r in iter_array(tables / 'ego_pose.json') if r['token'] in wanted_ego}
    times = np.arange(1, 65) * .1
    curves = [query_curve(float(v) * 15, float(k) * .03) for v, k in queries]
    summaries = {}; provenance = []
    for name, records in zip(['development', 'previously_used_holdout'], collections):
        labels = []; predictions = {m: [] for m in ['pinhole_static', 'oracle_future_frozen', 'privileged_past_static', 'privileged_past_cv']}
        scenes = []; eligible = []; motion_inputs = []; futures = []; object_indices = []; logs = []
        for object_index, obj, r in records:
            scene = r['scene']; input_sd = sd[ranked[scene]['sd_token']]
            state = past_state(tracks[r['instance_token']], samples, input_sd['timestamp'], egos[input_sd['ego_pose_token']])
            centers = np.array(r['agent_centers_xy']); sizes = np.array(r['agent_sizes']); yaws = np.array(r['agent_yaws'])
            cam = r['input_camera']; b = r['detection_box']; w, length, h = priors[r['label']]
            ray = rotation(cam['rotation']).apply(np.linalg.solve(np.array(cam['camera_intrinsic']), [(b[0]+b[2])/2, (b[1]+b[3])/2, 1]))
            depth = np.clip(cam['camera_intrinsic'][1][1] * h / max(b[3]-b[1], 1), 1, 180)
            center = np.array(cam['translation'][:2]) + depth * ray[:2]
            boxes = {'truth': (centers, yaws, sizes),
                     'pinhole_static': (np.tile(center, (64, 1)), np.zeros(64), np.tile([w, length, h], (64, 1))),
                     'oracle_future_frozen': (np.tile(centers[0], (64, 1)), np.repeat(yaws[0], 64), np.tile(sizes[0], (64, 1)))}
            if state is not None:
                c, velocity, yaw, size, prov = state
                boxes['privileged_past_static'] = (np.tile(c, (64, 1)), np.repeat(yaw, 64), np.tile(size, (64, 1)))
                boxes['privileged_past_cv'] = (c + times[:, None] * velocity, np.repeat(yaw, 64), np.tile(size, (64, 1)))
                provenance.append({'split': name, 'scene': scene, 'instance_token': r['instance_token'], **prov})
                numeric = np.r_[np.array(b) / [1600, 900, 1600, 900],
                                [r['label'] == label for label in priors], c, velocity,
                                np.cos(yaw), np.sin(yaw), size]
            else:
                numeric = np.zeros(4 + len(priors) + 2 + 2 + 2 + 3)
            motion_inputs.append(numeric); futures.append(centers)
            object_indices.append(object_index); logs.append(obj['log_token'])
            values = {}
            for method, (c, yaw, size) in boxes.items():
                distances = np.array([signed_point_box_distance(curve, c, yaw, size) for curve in curves])
                values[method] = np.stack([(distances[:, (times > lo) & (times <= hi + 1e-8)].min(-1) <= 2) for lo, hi in WINDOWS], -1)
            labels.append(values['truth']); scenes.append(scene); eligible.append(state is not None)
            for method in predictions:
                predictions[method].append(values.get(method, np.zeros_like(values['truth'])))
        y = np.array(labels); available = np.array(eligible); scenes = np.array(scenes)
        summaries[name] = {'agents': len(records), 'scenes': len(set(scenes)), 'agents_with_past': int(available.sum()), 'windows': {}}
        for j, window in enumerate(WINDOWS):
            row = {}
            # Every comparison uses exactly the agents with valid past, including static controls.
            for method, pp in predictions.items():
                pred = np.array(pp); yy = y[available, :, j]; p = pred[available, :, j]
                scene_y = np.stack([yy[scenes[available] == s].any(0) for s in sorted(set(scenes[available]))])
                scene_p = np.stack([p[scenes[available] == s].any(0) for s in sorted(set(scenes[available]))])
                row[method] = {'object': metrics(yy, p), 'scene_original_or': metrics(scene_y, scene_p)}
            summaries[name]['windows'][str(window)] = row
        np.savez_compressed(args.out / f'{name}_labels_predictions.npz', labels=y, eligible_past=available,
                            scenes=scenes, queries=queries, windows=np.array(WINDOWS),
                            numeric=np.array(motion_inputs), future_centers=np.array(futures),
                            object_indices=np.array(object_indices), logs=np.array(logs),
                            **{m: np.array(p) for m, p in predictions.items()})
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    (args.out / 'past_provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out / 'manifest.json').write_text(json.dumps({
        'status': 'EXPLORATORY, previously used development and holdout; no new untouched evaluation',
        'hypothesis': 'Whole-horizon OR labels hide temporal prediction errors; test time-window predicates before proposing temporal learning.',
        'windows_seconds': WINDOWS, 'queries': 'Unique speed-curvature pairs from both existing query modes, evaluated to 6.4s; not original OOD metric',
        'label': 'Within-2m centerline-to-box distance against logged nonreactive futures; not physical safety',
        'controls': 'Pinhole uses original numeric inputs. Both past controls use privileged 3D annotations. Future-frozen uses t+0.1s future geometry.',
        'comparison': 'Same past-eligible subset for all controls; fixed threshold distance <=2m, no fitting or selection',
        'data': [str(p) for p in datasets],
        'dataset_sha256': {str(p): hashlib.sha256((p/'dataset.npz').read_bytes()).hexdigest() for p in datasets},
    }, indent=2))
    print(json.dumps({name: {window: {m: round(v['scene_original_or']['balanced_error'], 4) if v['scene_original_or']['balanced_error'] is not None else None for m, v in row.items()} for window, row in s['windows'].items()} for name,s in summaries.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nuscenes', type=Path, default=Path('/mnt/datasets/nuScenes'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    run(args)


if __name__ == '__main__':
    main()
