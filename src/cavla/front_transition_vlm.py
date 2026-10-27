"""Single-frontal-image frozen ROI features for physical-transition information gate."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,AutoTokenizer
from cavla.multimodel_roi_features import load_backbone,snapshot
from cavla.object_features import roi_weights


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=Path('artifacts/front_transition_data_v1_20261021');rows=json.loads((source/'objects.json').read_text())
    model,path,loading=load_backbone(args.model)
    processor=AutoProcessor.from_pretrained(snapshot('Qwen/Qwen3-VL-2B-Instruct'),local_files_only=True,min_pixels=163840,max_pixels=196608)
    processor.tokenizer=AutoTokenizer.from_pretrained(snapshot('Qwen/Qwen3-VL-8B-Instruct'),local_files_only=True)
    features=np.zeros((len(rows),4,model.config.text_config.hidden_size),np.float32);scene_features=np.zeros_like(features)
    paths=sorted({r['image'] for r in rows});indices={path:[i for i,r in enumerate(rows) if r['image']==path] for path in paths}
    records={};start=time.monotonic();layers=[8,16,24,36]
    for done,ip in enumerate(paths):
        with Image.open(ip) as im:im=im.convert('RGB')
        messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'Describe the driving scene.'}]}]
        tokens=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        ids=tokens['input_ids'];positions=torch.where(ids[0]==model.config.image_token_id)[0]
        grid=tokens['image_grid_thw'][0];h,w=int(grid[1])//2,int(grid[2])//2
        assert len(positions)==h*w
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**tokens,output_hidden_states=True,use_cache=False,logits_to_keep=1)
            selected=torch.stack([output.hidden_states[layer][0,positions].float() for layer in layers])
            for i in indices[ip]:
                weights=torch.tensor(roi_weights(rows[i]['projected_box'],h,w),device='cuda',dtype=torch.float32)
                features[i]=(selected*weights[None,:,None]).sum(1).cpu().numpy()
                scene_features[i]=selected.mean(1).cpu().numpy()
        records[ip]=dict(grid=[h,w],input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),image_sha256=hashlib.sha256(Path(ip).read_bytes()).hexdigest(),tokens=int(ids.shape[1]),image_tokens=len(positions))
        del output,selected,tokens,ids
        if (done+1)%100==0:
            print(json.dumps(dict(model=args.model,done=done+1,total=len(paths),elapsed_seconds=time.monotonic()-start)),flush=True)
            (args.out/'progress.json').write_text(json.dumps(dict(done=done+1,total=len(paths))))
    np.savez_compressed(args.out/'features.npz',features=features,context=scene_features,layers=np.array(layers))
    (args.out/'manifest.json').write_text(json.dumps(dict(model=args.model,snapshot=str(path),loading=loading,objects_sha256=hashlib.sha256((source/'objects.json').read_bytes()).hexdigest(),
        scope='Frozen VLM backbone only; single frontal still, no policy head, no ego history',layers=layers,images=records,elapsed_seconds=time.monotonic()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))
    print('complete',args.model,len(rows),'agents',len(paths),'images',flush=True)

if __name__=='__main__':main()
