"""Past-only privileged map and surrounding-vehicle controls for visual onset probes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from matplotlib.path import Path as PolygonPath
from cavla.nuscenes_geometry import iter_array,rotation

LAYERS=['lane','carpark_area','drivable_area','walkway','stop_line','ped_crossing','intersection']


def distance_to_polygon(points,exterior,holes=()):
    points=np.asarray(points,float);rings=[np.asarray(exterior,float)]+[np.asarray(h,float) for h in holes]
    inside=PolygonPath(rings[0]).contains_points(points)
    for ring in rings[1:]:inside &= ~PolygonPath(ring).contains_points(points)
    result=np.full(len(points),np.inf)
    for ring in rings:
        aa=ring;bb=np.roll(ring,-1,axis=0);edge=bb-aa;norm=np.sum(edge**2,axis=1).clip(min=1e-16)
        for start in range(0,len(points),64):
            p=points[start:start+64];delta=p[:,None,:]-aa[None,:,:]
            progress=(np.sum(delta*edge[None,:,:],axis=2)/norm).clip(0,1)
            distance=np.linalg.norm(delta-progress[:,:,None]*edge[None,:,:],axis=2).min(1)
            result[start:start+len(p)]=np.minimum(result[start:start+len(p)],distance)
    return np.where(inside,-result,result)


def layer_distance(points,polygons,limit=50.):
    best=np.full(len(points),limit)
    for exterior,holes in polygons:
        lower=exterior.min(0);upper=exterior.max(0)
        bound=np.linalg.norm(np.maximum(np.maximum(lower-points,points-upper),0),axis=1)
        use=np.flatnonzero(bound<=np.maximum(best,0))
        if len(use):best[use]=np.minimum(best[use],distance_to_polygon(points[use],exterior,holes))
    return best.clip(-limit,limit)


def map_layers(path):
    m=json.loads(path.read_text());nodes={r['token']:np.array([r['x'],r['y']]) for r in m['node']}
    polygons={r['token']:(np.array([nodes[t] for t in r['exterior_node_tokens']]),[np.array([nodes[t] for t in h['node_tokens']]) for h in r['holes']]) for r in m['polygon']}
    result={}
    for layer in LAYERS:
        records=(m['lane']+m['lane_connector']) if layer=='lane' else ([r for r in m['road_segment'] if r['is_intersection']] if layer=='intersection' else m[layer])
        tokens=[token for r in records for token in (r['polygon_tokens'] if 'polygon_tokens' in r else [r['polygon_token']])]
        result[layer]=[polygons[t] for t in tokens]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=Path('artifacts/front_transition_egohistory_data_v1_20261021');rows=json.loads((source/'objects.json').read_text())
    tables=Path('/mnt/datasets/nuScenes/v1.0-trainval');root=tables.parent
    samples={r['token']:r for r in json.loads((tables/'sample.json').read_text())}
    logs={r['token']:r for r in json.loads((tables/'log.json').read_text())}
    instances={r['token']:r['category_token'] for r in json.loads((tables/'instance.json').read_text())}
    categories={r['token']:r['name'] for r in json.loads((tables/'category.json').read_text())}
    allowed={k for k,v in instances.items() if categories[v].startswith('vehicle.')}
    wanted={r['sample'] for r in rows};wanted |= {samples[t]['prev'] for t in list(wanted) if samples[t]['prev']}
    annotations={};by_sample={}
    for a in iter_array(tables/'sample_annotation.json'):
        if a['sample_token'] in wanted and a['instance_token'] in allowed:
            annotations[a['token']]={k:a[k] for k in ['token','prev','sample_token','instance_token','translation','rotation','size']}
            by_sample.setdefault(a['sample_token'],[]).append(a['token'])
    positions=[];neighbors=[];records=[]
    for r in rows:
        target=annotations[r['annotation']];at=samples[target['sample_token']]['timestamp'];assert at<=r['timestamp']
        age=(r['timestamp']-at)*1e-6;inv=rotation(target['rotation']).inv()
        def state(a):
            prev=annotations.get(a['prev']);position=np.array(a['translation'],float);velocity=np.zeros(3);known=False
            if prev is not None:
                gap=(samples[a['sample_token']]['timestamp']-samples[prev['sample_token']]['timestamp'])*1e-6
                if .25<=gap<=.75:velocity=(position-prev['translation'])/gap;known=True
            return position+age*velocity,velocity,known
        center,velocity,known=state(target);assert known
        group=[]
        for token in by_sample[target['sample_token']]:
            other=annotations[token]
            if other['instance_token']==target['instance_token']:continue
            oc,ov,ok=state(other);relative=inv.apply(oc-center);rel_velocity=inv.apply(ov-velocity)
            if np.linalg.norm(relative[:2])>100:continue
            yaw=(inv*rotation(other['rotation'])).as_euler('xyz')[2]
            values=np.r_[relative[:2],rel_velocity[:2],np.cos(yaw),np.sin(yaw),other['size'][:2],float(ok),1.]
            group.append((float(np.linalg.norm(relative[:2])),other['token'],values))
        group.sort(key=lambda x:(x[0],x[1]));features=np.zeros((8,10))
        for i,(_,_,values) in enumerate(group[:8]):features[i]=values
        positions.append(center[:2]);neighbors.append(features.ravel());records.append({'annotation':target['token'],'position_camera_extrapolated':center.tolist(),'neighbor_tokens':[x[1] for x in group[:8]],'past_timestamp':at})
    positions=np.array(positions);map_features=np.zeros((len(rows),len(LAYERS)*2));locations=np.array([logs[r['log']]['location'] for r in rows]);map_hashes={}
    for location in sorted(set(locations)):
        path=root/'maps/expansion'/f'{location}.json';layers=map_layers(path);use=np.flatnonzero(locations==location)
        map_hashes[location]=hashlib.sha256(path.read_bytes()).hexdigest()
        for j,name in enumerate(LAYERS):
            d=layer_distance(positions[use],layers[name]);map_features[use,j]=d;map_features[use,j+len(LAYERS)]=d<=0
        print('map',location,len(use),'complete',flush=True)
    with np.load(source/'features.npz') as z:arrays={k:z[k] for k in z.files}
    arrays['relative_geometry']=np.c_[arrays['relative_geometry'],map_features,np.array(neighbors)]
    np.savez_compressed(args.out/'features.npz',**arrays);(args.out/'objects.json').write_bytes((source/'objects.json').read_bytes())
    (args.out/'records.json').write_text(json.dumps(records,indent=2));(args.out/'manifest.json').write_text(json.dumps({'parent':str(source),
        'objects_sha256':hashlib.sha256((source/'objects.json').read_bytes()).hexdigest(),'map_hashes':map_hashes,'layers':LAYERS,
        'neighbor_dimensions':80,'map_dimensions':14,'scope':'Privileged past annotations and static map only; no attributes, future labels or learned detector',
        'limitations':['Straight CV extrapolation to camera <=0.75s','Polygon distance features do not certify road feasibility','Map constructed by dataset provider; no new independent split']},indent=2))
    print('complete',len(rows),'objects',flush=True)


if __name__=='__main__':main()
