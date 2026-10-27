"""NAVSIM-style non-reactive safety metrics for planned trajectories on PhysicalAI-AV clips.

Other agents follow their logged tracks (obstacle.offline autolabels), as in NAVSIM's
PDM score; there is no map, so drivable-area compliance is not measured. Boxes are given
in the rig frame at their own timestamps; each is moved to the world with the ego pose at
that timestamp, then into the ego frame at t0, and every track is interpolated onto the
10 Hz plan grid inside its observed time span. Metrics over the first 4 s:
collision (oriented-box overlap), near miss (box gap < 0.5 m), comfort violations with
NAVSIM thresholds, and progress along the plan.

Run with an environment that has physical_ai_av (the isolated Qwen-Drive venv).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HORIZON=40;DT=.1;EGO_LENGTH=4.9;EGO_WIDTH=2.1;NEAR_MISS_M=.5
COMFORT={'lon_acc_min':-4.05,'lon_acc_max':2.4,'lat_acc':4.89,'jerk':4.13,'yaw_rate':.95,'yaw_acc':1.93}
CLASSES={'automobile','person','trailer','bus','truck','bicycle','motorcycle','rider','heavy_truck','other_vehicle'}


def corners(x,y,yaw,length,width):
    c,s=np.cos(yaw),np.sin(yaw);dx=np.array([1,1,-1,-1])*length/2;dy=np.array([1,-1,-1,1])*width/2
    return np.stack([x[...,None]+c[...,None]*dx-s[...,None]*dy,y[...,None]+s[...,None]*dx+c[...,None]*dy],-1)


def overlap(a,b):
    """Separating-axis test for batches of convex quadrilaterals [...,4,2]."""
    result=np.ones(a.shape[:-2],bool)
    for poly in (a,b):
        for i in range(4):
            edge=poly[...,(i+1)%4,:]-poly[...,i,:];axis=np.stack([-edge[...,1],edge[...,0]],-1)
            pa=np.einsum('...kd,...d->...k',a,axis);pb=np.einsum('...kd,...d->...k',b,axis)
            result&=~((pa.max(-1)<pb.min(-1))|(pb.max(-1)<pa.min(-1)))
    return result


def gap(a,b):
    """Lower bound on the distance between two boxes: corner-to-corner minimum, 0 when overlapping."""
    d=np.linalg.norm(a[...,:,None,:]-b[...,None,:,:],axis=-1).min((-1,-2))
    return np.where(overlap(a,b),0.,d)


def headings(xy):
    d=np.diff(np.r_[[[0.,0.]],xy],axis=0);h=np.arctan2(d[:,1],d[:,0])
    moving=np.linalg.norm(d,axis=1)>.05
    for i in range(len(h)):
        if not moving[i]:h[i]=h[i-1] if i else 0.
    return h


def agents_in_ego_frame(avdi,clip,t0):
    try:df=avdi.get_clip_feature(clip,avdi.features.LABELS.OBSTACLE_OFFLINE,maybe_stream=True)['obstacle.offline']
    except KeyError:return None  # ~2.6% of clips ship without obstacle autolabels
    df=df[df.label_class.isin(CLASSES)]
    ego=avdi.get_clip_feature(clip,avdi.features.LABELS.EGOMOTION,maybe_stream=True)
    lo,hi=t0-200_000,t0+int((HORIZON+1)*DT*1e6)
    df=df[(df.timestamp_us>=lo)&(df.timestamp_us<=hi)]
    if df.empty:return []
    poses=ego(df.timestamp_us.to_numpy());now=ego(np.array([t0]));inverse=now.pose.rotation[0].inv();origin=now.pose.translation[0]
    centers=df[['center_x','center_y','center_z']].to_numpy()
    world=poses.pose.rotation.apply(centers)+poses.pose.translation
    local=inverse.apply(world-origin)
    box_rot=Rotation.from_quat(df[['orientation_x','orientation_y','orientation_z','orientation_w']].to_numpy())
    yaw=np.array([(inverse*r_ego*r_box).as_euler('xyz')[2] for r_ego,r_box in zip(poses.pose.rotation,box_rot)])
    grid=t0+np.round(np.arange(1,HORIZON+1)*DT*1e6);tracks=[]
    df=df.assign(x=local[:,0],y=local[:,1],yaw=np.unwrap(yaw))
    for _,track in df.sort_values('timestamp_us').groupby('track_id'):
        t=track.timestamp_us.to_numpy();inside=(grid>=t.min())&(grid<=t.max())
        if inside.sum()==0:continue
        x=np.interp(grid,t,track.x);y=np.interp(grid,t,track.y);h=np.interp(grid,t,np.unwrap(track.yaw.to_numpy()))
        tracks.append({'x':x,'y':y,'yaw':h,'valid':inside,'length':float(track.size_x.median()),'width':float(track.size_y.median()),
            'class':track.label_class.iloc[0]})
    return tracks


def score(plan,tracks,speed0):
    xy=np.asarray(plan,float)[:HORIZON,:2];h=headings(xy);ego=corners(xy[:,0],xy[:,1],h,EGO_LENGTH,EGO_WIDTH)
    collision=False;min_gap=np.inf
    for a in tracks:
        box=corners(a['x'],a['y'],a['yaw'],a['length'],a['width']);g=gap(ego,box)[a['valid']]
        if g.size:min_gap=min(min_gap,float(g.min()));collision|=bool((g==0).any())
    v=np.linalg.norm(np.diff(np.r_[[[0.,0.]],xy],axis=0),axis=1)/DT;v[0]=speed0 if np.isfinite(speed0) else v[0]
    acc=np.diff(v)/DT;jerk=np.diff(acc)/DT;yaw_rate=np.diff(np.unwrap(h))/DT;yaw_acc=np.diff(yaw_rate)/DT
    lat=v[1:]*yaw_rate
    comfort_ok=bool((acc.min()>=COMFORT['lon_acc_min'])&(acc.max()<=COMFORT['lon_acc_max'])&(np.abs(lat).max()<=COMFORT['lat_acc'])
        &(np.abs(jerk).max()<=COMFORT['jerk'])&(np.abs(yaw_rate).max()<=COMFORT['yaw_rate'])&(np.abs(yaw_acc).max()<=COMFORT['yaw_acc']))
    return {'collision':collision,'near_miss':bool(min_gap<NEAR_MISS_M),'min_gap':min_gap if np.isfinite(min_gap) else None,
        'comfort_ok':comfort_ok,'progress_m':float(np.linalg.norm(np.diff(np.r_[[[0.,0.]],xy],axis=0),axis=1).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True)
    p.add_argument('--probes',type=Path,nargs='+',required=True,help='Directories with plans.npz from temperature probes')
    p.add_argument('--human-from',type=Path,default=None,help='Context npz directory with ego_future_xyz when a probe lacks *_human')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import physical_ai_av
    avdi=physical_ai_av.PhysicalAIAVDatasetInterface()
    records={r['name']:r for r in json.loads(args.records.read_text()) if 'error' not in r}
    plans={probe.name:dict(np.load(probe/'plans.npz')) for probe in args.probes}
    names=sorted({k.split('_s')[0] for p_ in plans.values() for k in p_ if '_s92' in k})
    rows=[];missing=[]
    for i,name in enumerate(names):
        record=records[name];tracks=agents_in_ego_frame(avdi,record['clip_id'],record['t0_us']);speed=record['speed_mps']
        if tracks is None:
            missing.append(name);print(json.dumps({'context':name,'skipped':'no obstacle labels'}),flush=True);continue
        human=None
        for probe_plans in plans.values():
            if f'{name}_human' in probe_plans:human=probe_plans[f'{name}_human']
        if human is None and args.human_from:human=np.load(args.human_from/f'{name}.npz')['ego_future_xyz']
        if human is not None:rows.append({'probe':'human','context':name,'seed':0,'config':'logged',**score(human,tracks,speed),'agents':len(tracks)})
        for probe,probe_plans in plans.items():
            for key,plan in probe_plans.items():
                if not key.startswith(name+'_s92'):continue
                seed=int(key[len(name)+2:len(name)+6]);config=key.split('_',3)[-1]
                rows.append({'probe':probe,'context':name,'seed':seed,'config':config,**score(plan,tracks,speed),'agents':len(tracks)})
        print(json.dumps({'context':name,'agents':len(tracks),'done':i+1,'of':len(names)}),flush=True)
        (args.out/'rows.json').write_text(json.dumps(rows))
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'probes':[str(p_) for p_ in args.probes],'skipped_no_labels':missing,'horizon_steps':HORIZON,
        'ego_box_m':[EGO_LENGTH,EGO_WIDTH],'near_miss_m':NEAR_MISS_M,'comfort':COMFORT,'classes':sorted(CLASSES),
        'limitations':['Non-reactive log replay of other agents','No map: drivable-area compliance not measured',
            'Fixed ego box; heading from plan differences','Autolabel tracks interpolated only inside their observed span']},indent=2))


if __name__=='__main__':main()
