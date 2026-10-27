"""Read-only provenance and image diagnostics; never generate semantic labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image


def save_json(path, value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def unique_box(stem, metadata, width, height):
    match=re.fullmatch(r'\d+_(.+)_conf([0-9.]+)',stem)
    if not match:return None
    label,confidence=match.group(1),float(match.group(2))
    boxes=[r['box_xyxy'] for r in metadata.get('objects',[])
           if r.get('label')==label and abs(r.get('confidence',-1)-confidence)<=.00501]
    if len(boxes)!=1:return None
    x0,y0,x1,y1=boxes[0]
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):return None
    return boxes[0]


def edit_diagnostic(original, edited, box):
    if original.shape!=edited.shape:return dict(shape_match=False)
    diff=np.abs(original.astype(np.int16)-edited.astype(np.int16)).mean(axis=-1)
    result=dict(shape_match=True,mean_abs_rgb=float(diff.mean()),pixel_fraction_above_8=float((diff>8).mean()))
    if box is None:return result
    x0,y0,x1,y1=box;h,w=diff.shape
    mask=np.ones((h,w),dtype=bool)
    mask[max(0,int(y0)-8):min(h,int(np.ceil(y1))+8),max(0,int(x0)-8):min(w,int(np.ceil(x1))+8)]=False
    result.update(outside_box_mean_abs_rgb=float(diff[mask].mean()) if mask.any() else 0.,
                  outside_box_pixel_fraction_above_8=float((diff[mask]>8).mean()) if mask.any() else 0.,
                  outside_box_change_mass_fraction=float(diff[mask].sum()/max(diff.sum(),1e-12)))
    return result


def read_annotations(path):
    annotations={}
    for p in sorted(path.glob('*.csv')):
        with p.open(newline='',encoding='utf-8-sig') as f:
            for raw in csv.DictReader(f):
                row={k.strip():v for k,v in raw.items() if isinstance(k,str)}
                try: scene=f"scene-{int(row.get('Scene Number','').strip()):04d}"
                except ValueError:continue
                text=(row.get('Instruction') or '').strip()
                if text: annotations.setdefault(scene,{})[text]=row.get('Instruction Type','').strip()
    return annotations


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(args):
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('Fresh output required')
    args.out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__);shutil.copyfile(source,args.out/source.name)
    ranked={r['scene_name']:r for r in csv.DictReader(args.ranked.open())}
    records=list(csv.DictReader(args.pairs.open()))
    annotations=read_annotations(args.annotations)
    nusc_scenes={r['token']:r for r in json.loads((args.nuscenes/'v1.0-trainval/scene.json').read_text())}
    samples={r['token']:r for r in json.loads((args.nuscenes/'v1.0-trainval/sample.json').read_text())}
    scene_rows={}
    for row in records:scene_rows.setdefault(row['scene'],[]).append(row)
    save_json(args.out/'manifest.json',dict(scope='Provenance and approximate edit diagnostics; no semantic ground truth',
        config={k:str(v) for k,v in vars(args).items()},source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.ranked,args.pairs]},
        bbox_join='Unique label/confidence within 0.00501, never ordinal alone; still not verified mask identity',
        source_image_match_threshold='Mean absolute decoded RGB difference <=2 on 0..255; diagnostic, not exact identity proof',
        annotation_status='All task/reference/action labels left empty; only scene-level candidate joins'))

    def process(item):
        scene,rows=item;folder=args.images/scene;meta=ranked.get(scene)
        frame=dict(scene=scene,ranked_record_present=meta is not None)
        if meta is None:return frame,[dict(scene=scene,stem=r['stem'],status='missing_provenance') for r in rows]
        sample=samples.get(meta['sample_token'])
        frame.update(sample_token=meta['sample_token'],sd_token=meta['sd_token'],timestamp_us=meta['timestamp_us'],
            sample_matches_scene=bool(sample and nusc_scenes[sample['scene_token']]['name']==scene),
            instruction_count=len(annotations.get(scene,{})))
        original_path=folder/f'{scene}_front.jpg'
        source_path=args.nuscenes/'samples/CAM_FRONT'/Path(meta['image_path']).name
        frame.update(original_path=str(original_path),source_path=str(source_path),source_image_exists=source_path.exists())
        with Image.open(original_path) as im:original=np.array(im.convert('RGB'))
        if source_path.exists():
            with Image.open(source_path) as im:reference=np.array(im.convert('RGB'))
            frame['source_dimensions_match']=original.shape==reference.shape
            frame['source_mean_abs_rgb']=float(np.abs(original.astype(np.int16)-reference.astype(np.int16)).mean()) if original.shape==reference.shape else None
        with np.load(folder/f'{scene}_data.npz',allow_pickle=False) as data:
            saved=data['ego_history_xyz']
            expected=np.array(json.loads(meta['ego_history_xyz']),dtype=np.float32)
            frame['history_matches_provenance']=saved.shape==expected.shape and bool(np.allclose(saved,expected,atol=1e-6))
        metadata=json.loads((folder/f'{scene}_seg.json').read_text())
        result=[];h,w=original.shape[:2]
        for row in rows:
            stem=row['stem'];path=folder/'inpainted_clean_fin'/f'{stem}_imagen.png'
            out=dict(scene=scene,stem=stem,label=row['label'],image_exists=path.exists(),edited_path=str(path))
            if path.exists():
                with Image.open(path) as im:edited=np.array(im.convert('RGB'))
                box=unique_box(stem,metadata,w,h);out['unique_bbox_match']=box is not None
                out.update(edit_diagnostic(original,edited,box))
            result.append(out)
        return frame,result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        processed=list(executor.map(process,sorted(scene_rows.items())))
    frames=[x[0] for x in processed];edits=[r for x in processed for r in x[1]]
    write_csv(args.out/'frames.csv',frames);write_csv(args.out/'edits.csv',edits)
    usable_scenes={f['scene'] for f in frames if f.get('sample_matches_scene') and f.get('history_matches_provenance')
                   and f.get('source_mean_abs_rgb') is not None and f['source_mean_abs_rgb']<=2 and f.get('instruction_count',0)>0}
    eligible={}
    for r in edits:
        if r['scene'] in usable_scenes and r.get('shape_match') and r.get('unique_bbox_match'):eligible.setdefault(r['scene'],[]).append(r)
    rng=np.random.default_rng(20261020);chosen=rng.permutation(sorted(eligible))[:60]
    candidates=[]
    for scene in chosen:
        row=eligible[scene][int(rng.integers(len(eligible[scene])))]
        texts=sorted(annotations[scene]);text=texts[int(rng.integers(len(texts)))]
        candidates.append(dict(scene=scene,stem=row['stem'],instruction=text,instruction_type=annotations[scene][text],
            sample_token=ranked[scene]['sample_token'],original_path=str(args.images/scene/f'{scene}_front.jpg'),
            edited_path=row['edited_path'],temporal_alignment_verified='',referent_identity_verified='',
            intervention_validity='',task_valid_before='',task_valid_after='',acceptable_action_constraint='',
            annotator_id='',uncertainty_or_abstention='',status='UNLABELLED_CANDIDATE'))
    write_csv(args.out/'annotation_candidates.csv',candidates)
    with_box=[r for r in edits if 'outside_box_pixel_fraction_above_8' in r]
    summary=dict(n_scenes=len(frames),n_pairs=len(edits),source_images_present=sum(f.get('source_image_exists',False) for f in frames),
        sample_scene_matches=sum(f.get('sample_matches_scene',False) for f in frames),
        history_matches=sum(f.get('history_matches_provenance',False) for f in frames),
        source_rgb_difference_median=float(np.median([f['source_mean_abs_rgb'] for f in frames if f.get('source_mean_abs_rgb') is not None])),
        existing_edit_images=sum(r.get('image_exists',False) for r in edits),bbox_diagnostic_pairs=len(with_box),
        outside_box_fraction_above8_median=float(np.median([r['outside_box_pixel_fraction_above_8'] for r in with_box])),
        pairs_with_over_1pct_pixels_changed_outside_box=sum(r['outside_box_pixel_fraction_above_8']>.01 for r in with_box),
        scenes_with_provenance_and_any_instruction=len(usable_scenes),unlabelled_annotation_candidates=len(candidates),
        limitations=['No verified referents or normative counterfactual actions',
                     'Bounding boxes are approximate joins, not verified segmentation masks',
                     'Off-box differences may be legitimate boundary changes or artifacts; not a semantic-validity label'])
    save_json(args.out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,default=Path('artifacts/real_audit_20261020/pairs.csv'))
    parser.add_argument('--ranked',type=Path,default=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/scenes_ranked.csv'))
    parser.add_argument('--images',type=Path,default=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene'))
    parser.add_argument('--annotations',type=Path,default=Path('/mnt/datasets/doScenes/Annotations'))
    parser.add_argument('--nuscenes',type=Path,default=Path('/mnt/datasets/nuScenes'))
    parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args())
