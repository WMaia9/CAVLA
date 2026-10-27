"""Merge per-episode CARLA collections into one flat solver-probe input folder (symlinks only)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    records=[];index=0;episodes=sorted(d for d in args.root.glob('ep_*') if d.is_dir())
    for folder in episodes:
        manifest=json.loads((folder/'manifest.json').read_text())
        for record in json.loads((folder/'records.json').read_text()):
            source=Path(record['path']).resolve();name=f'context_{index:03d}'
            if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError(f'Hash mismatch {source}')
            (args.out/f'{name}.npz').symlink_to(source)
            records.append({'name':name,'episode':folder.name,'map':manifest['map'],'spawn_index':manifest['spawn_index'],
                'target_speed':manifest['target_speed'],'speed_mps':record['speed_mps'],'source':str(source),'sha256':record['sha256']})
            index+=1
    (args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'episodes':[d.name for d in episodes],'contexts':len(records),
        'limitations':['Contexts within an episode are correlated','Scripted lane-following, static NPCs']},indent=2))
    print(json.dumps({'episodes':len(episodes),'contexts':len(records)}))


if __name__=='__main__':main()
