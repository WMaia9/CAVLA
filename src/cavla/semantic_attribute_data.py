"""Past-timestamp nuScenes semantic attributes for an exploratory intent gate.

Attributes are human annotation proxies, not observed driver mental states.
No post-input annotation is used as an input attribute.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cavla.nuscenes_geometry import iter_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    source = Path('artifacts/temporal_necessity_v2_20261021')
    provenance = json.loads((source/'past_provenance.json').read_text())
    wanted = {(r['instance_token'], r['latest_annotation_timestamp']) for r in provenance}
    tables = Path('/mnt/datasets/nuScenes/v1.0-trainval')
    samples = {r['token']: r['timestamp'] for r in json.loads((tables/'sample.json').read_text())}
    names = {r['token']: r['name'] for r in json.loads((tables/'attribute.json').read_text())}
    annotations = {}
    for ann in iter_array(tables/'sample_annotation.json'):
        key = ann['instance_token'], samples[ann['sample_token']]
        if key in wanted:
            annotations[key] = ann
    info = {(r['split'], r['scene'], r['instance_token']): r for r in provenance}
    summaries = {}; records = []
    for split, data, geometry in [
        ('development', 'partial_composition_data_v2_20261021', 'nuscenes_geometry_v3_20261021'),
        ('previously_used_holdout', 'composition_holdout_data_v1_20261021', 'composition_holdout_geometry_v1_20261021')]:
        objects = json.loads((Path('artifacts')/data/'objects.json').read_text())
        geom = {(r['scene'], r['stem']): r for r in json.loads((Path('artifacts')/geometry/'records.json').read_text())}
        with np.load(source/f'{split}_labels_predictions.npz') as z:
            d = {k: z[k][z['eligible_past']] for k in ['object_indices', 'numeric', 'future_centers', 'logs', 'scenes']}
        attrs = []
        for i, objidx in enumerate(d['object_indices']):
            obj = objects[objidx]; g = geom[obj['scene'], obj['stem']]
            prov = info[split, obj['scene'], g['instance_token']]
            assert prov['latest_annotation_timestamp'] <= prov['input_timestamp']
            ann = annotations[g['instance_token'], prov['latest_annotation_timestamp']]
            labels = [names[t] for t in ann['attribute_tokens']]
            attrs.append(labels[0] if len(labels) == 1 else 'unknown')
            records.append({'split': split, 'scene': obj['scene'], 'stem': obj['stem'], 'label': g['label'],
                            'attributes': labels, 'annotation_token': ann['token'],
                            'attribute_timestamp': prov['latest_annotation_timestamp'], 'input_timestamp': prov['input_timestamp']})
        attrs = np.array(attrs)
        speed = np.linalg.norm(d['numeric'][:, 12:14], axis=-1)
        displacement = np.linalg.norm(d['future_centers'] - d['numeric'][:, None, 10:12], axis=-1).max(1)
        slow_vehicle = (speed <= .5) & np.isin(attrs, ['vehicle.parked', 'vehicle.stopped'])
        event = displacement > 2.
        summary = {'all_agents': len(attrs), 'past_attribute_counts': {a: int((attrs == a).sum()) for a in sorted(set(attrs))},
                   'slow_parked_stopped_agents': int(slow_vehicle.sum()), 'slow_scenes': len(set(d['scenes'][slow_vehicle])),
                   'slow_logs': len(set(d['logs'][slow_vehicle])), 'future_departure_events': int(event[slow_vehicle].sum()),
                   'slow_attribute_departure_counts': {a: {'agents': int(((attrs == a) & slow_vehicle).sum()),
                       'departure': int((event & (attrs == a) & slow_vehicle).sum())} for a in ['vehicle.parked', 'vehicle.stopped']}}
        summaries[split] = summary
        np.savez_compressed(args.out/f'{split}.npz', **d, attributes=attrs, past_speed=speed,
                            future_displacement=displacement, slow_vehicle=slow_vehicle, departure=event)
    (args.out/'summary.json').write_text(json.dumps(summaries, indent=2))
    (args.out/'provenance.json').write_text(json.dumps(records, indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'manifest.json').write_text(json.dumps({'scope': 'Exploratory data sufficiency gate, existing evaluation already used',
        'selection': 'Past speed <=0.5 m/s and past human attribute parked/stopped, fixed before looking at counts',
        'event': 'Logged maximum displacement >2m from past-extrapolated current center within6.4s; not ego safety or counterfactual response',
        'limitations': ['Attribute timestamps precede input, but retrospective annotation procedures may use future information',
                        'Not verified driver mental intent', 'Automatic projection associations', 'No independent holdout']}, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
