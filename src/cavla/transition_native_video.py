"""Native Qwen3-VL video encoder gate, paired with a repeated-current-video control."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image


def prepare(processor,frames,mode):
    from transformers.video_utils import VideoMetadata
    paths=[f['image'] if mode=='temporal' else frames[-1]['image'] for f in frames]
    video=[]
    for path in paths:
        with Image.open(path) as im:video.append(np.asarray(im.convert('RGB')))
    times=[f['timestamp'] for f in frames];offsets=[t-times[0] for t in times]
    assert len(video)==4 and all(t<=times[-1] for t in times)
    # Processor only divides indices by fps to get timestamps; sampling is disabled.
    # These are integer microsecond coordinates, NOT a claim of MHz camera acquisition.
    metadata=VideoMetadata(total_num_frames=offsets[-1]+1,fps=1e6,width=video[0].shape[1],height=video[0].shape[0],
        duration=offsets[-1]*1e-6,frames_indices=offsets)
    messages=[{'role':'user','content':[{'type':'video'},{'type':'text','text':'Describe the driving scene.'}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    tokens=processor(text=[text],videos=[np.stack(video)],return_tensors='pt',
        videos_kwargs={'do_sample_frames':False,'video_metadata':[metadata],'size':{'shortest_edge':4*163840,'longest_edge':4*196608}})
    assert tokens['video_grid_thw'].shape==(1,3) and int(tokens['video_grid_thw'][0,0])==2
    assert metadata.frames_indices==offsets,'Processor changed the selected physical frames'
    return tokens,paths,{'timestamps_microseconds':times,'offsets_microseconds':offsets,'do_sample_frames':False,'timestamp_encoding':'Integer microseconds / 1e6; not acquisition FPS'}


def main():
    import torch
    from transformers import AutoProcessor,AutoTokenizer
    from cavla.multimodel_roi_features import load_backbone,snapshot
    from cavla.object_features import roi_weights
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['temporal','repeat'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);source_bytes=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source_bytes);torch.set_num_threads(2)
    source=Path('artifacts/front_transition_data_v1_20261021/objects.json');rows=json.loads(source.read_text());histories=json.loads(Path('artifacts/transition_visual_history_data_v1_20261021/histories.json').read_text())
    processor=AutoProcessor.from_pretrained(snapshot('Qwen/Qwen3-VL-2B-Instruct'),local_files_only=True,min_pixels=163840,max_pixels=196608)
    processor.tokenizer=AutoTokenizer.from_pretrained(snapshot('Qwen/Qwen3-VL-8B-Instruct'),local_files_only=True)
    if args.check_only:
        frames=histories[sorted(histories)[0]];a,_,meta=prepare(processor,frames,'temporal');b,_,_=prepare(processor,frames,'repeat')
        assert torch.equal(a['input_ids'],b['input_ids']) and torch.equal(a['video_grid_thw'],b['video_grid_thw'])
        assert not torch.equal(a['pixel_values_videos'],b['pixel_values_videos'])
        expected=(int(a['video_grid_thw'].prod())//4);assert int((a['input_ids']==processor.video_token_id).sum())==expected
        summary={'grid':a['video_grid_thw'].tolist(),'tokens':int(a['input_ids'].shape[1]),'video_tokens':expected,
            'input_ids_equal':True,'pixels_differ':True,'metadata':meta,'decoded_text':processor.tokenizer.decode(a['input_ids'][0])[:160]}
        (args.out/'processor_check.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));return
    model,path,loading=load_backbone('Qwen/Qwen3-VL-8B-Instruct');indices=defaultdict(list)
    for i,r in enumerate(rows):indices[r['image']].append(i)
    features=np.zeros((len(rows),1,model.config.text_config.hidden_size),np.float32);context=np.zeros_like(features);records={};hashes={};start=time.monotonic()
    for done,ip in enumerate(sorted(indices)):
        tokens,paths,meta=prepare(processor,histories[ip],args.mode);tokens=tokens.to('cuda');ids=tokens['input_ids'];grid=tokens['video_grid_thw'][0]
        h,w=int(grid[1])//2,int(grid[2])//2;positions=torch.where(ids[0]==model.config.video_token_id)[0]
        assert len(positions)==2*h*w;positions=positions[-h*w:]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**tokens,output_hidden_states=True,use_cache=False,logits_to_keep=1);selected=output.hidden_states[36][0,positions].float()
            for i in indices[ip]:
                weights=torch.tensor(roi_weights(rows[i]['projected_box'],h,w),device='cuda',dtype=torch.float32)
                features[i,0]=(selected*weights[:,None]).sum(0).cpu().numpy();context[i,0]=selected.mean(0).cpu().numpy()
        for path_used in paths:
            if path_used not in hashes:hashes[path_used]=hashlib.sha256(Path(path_used).read_bytes()).hexdigest()
        records[ip]={'grid':grid.cpu().tolist(),'tokens':int(ids.shape[1]),'input_ids_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),'paths':paths,**meta}
        del output,selected,tokens,ids
        if (done+1)%100==0:
            print(json.dumps({'mode':args.mode,'done':done+1,'total':len(indices),'elapsed_seconds':time.monotonic()-start}),flush=True)
            (args.out/'progress.json').write_text(json.dumps({'done':done+1,'total':len(indices)}))
    np.savez_compressed(args.out/'features.npz',features=features,context=context,layers=np.array([36]))
    (args.out/'manifest.json').write_text(json.dumps({'model':'Qwen/Qwen3-VL-8B-Instruct','mode':args.mode,'snapshot':str(path),'loading':loading,
        'objects_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'images':records,'image_hashes':hashes,
        'scope':'Native four-frame video; last two-frame temporal patch ROI/context; frozen backbone, no planner',
        'elapsed_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))
    print('complete',args.mode,flush=True)


if __name__=='__main__':main()
