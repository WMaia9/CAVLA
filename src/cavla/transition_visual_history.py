"""Past-only four-keyframe inventory and matched-budget frozen visual histories."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from cavla.nuscenes_geometry import iter_array


def inventory(out):
    out.mkdir(parents=True,exist_ok=False);(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=Path('artifacts/front_transition_data_v1_20261021/objects.json');rows=json.loads(source.read_text())
    tables=Path('/mnt/datasets/nuScenes/v1.0-trainval');root=tables.parent
    samples={r['token']:r for r in json.loads((tables/'sample.json').read_text())}
    sensors={r['token']:r for r in json.loads((tables/'sensor.json').read_text())}
    calibrated={r['token'] for r in json.loads((tables/'calibrated_sensor.json').read_text()) if sensors[r['sensor_token']]['channel']=='CAM_FRONT'}
    scenes=defaultdict(list)
    for r in iter_array(tables/'sample_data.json'):
        if r['is_key_frame'] and r['calibrated_sensor_token'] in calibrated and (root/r['filename']).is_file():
            scenes[samples[r['sample_token']]['scene_token']].append({'token':r['token'],'timestamp':r['timestamp'],'image':str(root/r['filename'])})
    histories={};missing=[]
    for image in sorted({r['image'] for r in rows}):
        row=next(r for r in rows if r['image']==image);scene=samples[row['camera_sample']]['scene_token']
        frames=sorted(scenes[scene],key=lambda r:r['timestamp']);index=next(i for i,r in enumerate(frames) if r['token']==row['camera_token'])
        if index<3:missing.append(image);continue
        frames=frames[index-3:index+1];times=np.array([r['timestamp'] for r in frames])
        if np.any((np.diff(times)*1e-6<.25)|(np.diff(times)*1e-6>.75)):missing.append(image);continue
        assert frames[-1]['image']==image and times[-1]==row['timestamp']
        assert all(t<=row['timestamp'] for t in times)
        histories[image]=frames
    (out/'histories.json').write_text(json.dumps(histories,indent=2))
    (out/'manifest.json').write_text(json.dumps({'objects_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'latest_images':len({r['image'] for r in rows}),'histories':len(histories),'missing':missing,
        'selection':'Four most recent same-scene frontal keyframes ending at selected camera; no future or event filter'},indent=2))
    print('histories',len(histories),'missing',len(missing),flush=True)


def extract(mode,out):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoTokenizer
    from cavla.multimodel_roi_features import load_backbone,snapshot
    from cavla.object_features import roi_weights
    out.mkdir(parents=True,exist_ok=False);(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());torch.set_num_threads(2)
    source=Path('artifacts/front_transition_data_v1_20261021/objects.json');rows=json.loads(source.read_text())
    hdir=Path('artifacts/transition_visual_history_data_v1_20261021');histories=json.loads((hdir/'histories.json').read_text())
    assert not json.loads((hdir/'manifest.json').read_text())['missing'],'Paired cohort must be specified if any history unavailable'
    model,path,loading=load_backbone('Qwen/Qwen3-VL-8B-Instruct')
    processor=AutoProcessor.from_pretrained(snapshot('Qwen/Qwen3-VL-2B-Instruct'),local_files_only=True,min_pixels=163840,max_pixels=196608)
    processor.tokenizer=AutoTokenizer.from_pretrained(snapshot('Qwen/Qwen3-VL-8B-Instruct'),local_files_only=True)
    features=np.zeros((len(rows),1,model.config.text_config.hidden_size),np.float32);context=np.zeros_like(features)
    indices=defaultdict(list)
    for i,r in enumerate(rows):indices[r['image']].append(i)
    records={};hashes={};start=time.monotonic()
    for done,ip in enumerate(sorted(indices)):
        frames=histories[ip];content=[];latest=frames[-1]['timestamp'];paths=[]
        for frame in frames:
            path_used=frame['image'] if mode=='temporal' else ip;paths.append(path_used)
            content.append({'type':'text','text':f"Time {(frame['timestamp']-latest)*1e-6:.3f} seconds:"})
            with Image.open(path_used) as im:im=im.convert('RGB')
            content.append({'type':'image','image':im})
            if path_used not in hashes:hashes[path_used]=hashlib.sha256(Path(path_used).read_bytes()).hexdigest()
        content.append({'type':'text','text':'Describe the driving scene.'})
        messages=[{'role':'user','content':content}]
        tokens=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        ids=tokens['input_ids'];positions=torch.where(ids[0]==model.config.image_token_id)[0]
        grids=tokens['image_grid_thw'];assert len(grids)==4
        counts=[(int(g[1])//2)*(int(g[2])//2) for g in grids];assert len(positions)==sum(counts)
        h,w=int(grids[-1,1])//2,int(grids[-1,2])//2;positions=positions[-counts[-1]:]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**tokens,output_hidden_states=True,use_cache=False,logits_to_keep=1)
            selected=output.hidden_states[36][0,positions].float()
            for i in indices[ip]:
                weights=torch.tensor(roi_weights(rows[i]['projected_box'],h,w),device='cuda',dtype=torch.float32)
                features[i,0]=(selected*weights[:,None]).sum(0).cpu().numpy();context[i,0]=selected.mean(0).cpu().numpy()
        records[ip]={'input_ids_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),'grids':grids.cpu().tolist(),
            'tokens':int(ids.shape[1]),'image_tokens':counts,'paths':paths,'physical_timestamps':[f['timestamp'] for f in frames]}
        del output,selected,tokens,ids
        if (done+1)%100==0:
            print(json.dumps({'mode':mode,'done':done+1,'total':len(indices),'elapsed_seconds':time.monotonic()-start}),flush=True)
            (out/'progress.json').write_text(json.dumps({'done':done+1,'total':len(indices)}))
    np.savez_compressed(out/'features.npz',features=features,context=context,layers=np.array([36]))
    (out/'manifest.json').write_text(json.dumps({'model':'Qwen/Qwen3-VL-8B-Instruct','mode':mode,'snapshot':str(path),'loading':loading,
        'objects_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'history_manifest_sha256':hashlib.sha256((hdir/'histories.json').read_bytes()).hexdigest(),
        'scope':'Four still-image inputs with explicit past timestamps; frozen backbone only; no policy head or future images',
        'images':records,'image_hashes':hashes,'elapsed_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))
    print('complete',mode,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['inventory','temporal','repeat'],required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.mode=='inventory':inventory(args.out)
    else:extract(args.mode,args.out)


if __name__=='__main__':main()
