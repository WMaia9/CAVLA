"""Collect isolated CARLA four-camera contexts with explicit handedness conversion.

Infrastructure/diagnostic data only; scripted recordings are not counterfactual
references or evidence of an original learning method.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import queue
import random
import time
import traceback
import numpy as np

RIG = [('cross_left',1.,-.5,-60.,120),('front_wide',1.5,0.,0.,120),
       ('cross_right',1.,.5,60.,120),('front_tele',1.5,0.,0.,30)]


def local_right_handed_history(matrices):
    matrices = np.asarray(matrices,float); reflection = np.diag([1.,-1.,1.])
    rotations = reflection@matrices[:,:3,:3]@reflection
    positions = matrices[:,:3,3]@reflection
    local_xyz = (positions-positions[-1])@rotations[-1]
    local_rot = rotations[-1].T@rotations
    return local_xyz.astype(np.float32),local_rot.astype(np.float32)


def packet_at(sensor_queue,frame):
    while True:
        data = sensor_queue.get(timeout=30)
        if data.frame == frame: return data
        if data.frame > frame: raise RuntimeError(f'Sensor skipped frame {frame}, got {data.frame}')


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=2100);p.add_argument('--contexts',type=int,default=4)
    p.add_argument('--npcs',type=int,default=0)
    p.add_argument('--map',default=None);p.add_argument('--spawn-index',type=int,default=None)
    p.add_argument('--seed',type=int,default=20261022);p.add_argument('--target-speed',type=float,default=5.)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    client=carla.Client('127.0.0.1',args.port);client.set_timeout(60)
    world=client.load_world(args.map) if args.map else client.get_world()
    settings=world.get_settings();saved_settings=world.get_settings()
    actors=[];sensors=[];start=time.time();records=[]
    try:
        settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.substepping=True
        settings.max_substep_delta_time=.01;settings.max_substeps=10;world.apply_settings(settings)
        library=world.get_blueprint_library();spawns=world.get_map().get_spawn_points()
        ego=None
        order=spawns if args.spawn_index is None else spawns[args.spawn_index:]+spawns[:args.spawn_index]
        for transform in order:
            ego=world.try_spawn_actor(library.find('vehicle.tesla.model3'),transform)
            if ego:break
        if ego is None:raise RuntimeError('No ego spawn available')
        actors.append(ego)
        rng=random.Random(args.seed);candidates=list(spawns);rng.shuffle(candidates)
        blueprints=sorted(library.filter('vehicle.*'),key=lambda bp:bp.id)
        for transform in candidates:
            if len(actors)>=args.npcs+1:break
            npc=world.try_spawn_actor(rng.choice(blueprints),transform)
            if npc:actors.append(npc);npc.set_simulate_physics(False)
        queues=[]
        for name,x,y,yaw,fov in RIG:
            bp=library.find('sensor.camera.rgb');bp.set_attribute('image_size_x','1920');bp.set_attribute('image_size_y','1080')
            # Emit at every world tick. A nominal .1 sensor period can skip a
            # tick through float accumulation even when the world dt is .1.
            bp.set_attribute('fov',str(fov));bp.set_attribute('sensor_tick','0.0')
            sensor=world.spawn_actor(bp,carla.Transform(carla.Location(x=x,y=y,z=2.4),carla.Rotation(yaw=yaw)),attach_to=ego)
            sensors.append(sensor);q=queue.Queue();sensor.listen(q.put);queues.append(q)
        pose_history=deque(maxlen=16);image_history=deque(maxlen=4);frame_history=deque(maxlen=16)
        print(json.dumps({'stage':'sensors_ready','ego_id':ego.id,'npc_count':len(actors)-1}),flush=True)
        for tick in range(30+20*args.contexts):
            pose=ego.get_transform();velocity=ego.get_velocity();speed=np.linalg.norm([velocity.x,velocity.y,velocity.z])
            waypoint=world.get_map().get_waypoint(pose.location);next_waypoints=waypoint.next(8.)
            target=next_waypoints[0].transform.location if next_waypoints else pose.location+pose.get_forward_vector()*8
            desired=np.arctan2(target.y-pose.location.y,target.x-pose.location.x)
            heading=np.deg2rad(pose.rotation.yaw);error=(desired-heading+np.pi)%(2*np.pi)-np.pi
            ego.apply_control(carla.VehicleControl(throttle=float(np.clip(.3*(args.target_speed-speed),0,.6)),
                brake=float(np.clip(.2*(speed-args.target_speed),0,.5)),steer=float(np.clip(1.5*error,-.7,.7))))
            frame=world.tick();snapshot=world.get_snapshot();packets=[packet_at(q,frame) for q in queues]
            assert all(abs(packet.timestamp-snapshot.timestamp.elapsed_seconds)<1e-6 for packet in packets)
            images=np.stack([np.frombuffer(packet.raw_data,np.uint8).reshape(1080,1920,4)[:,:,:3][:,:,::-1].copy() for packet in packets])
            pose_history.append(np.array(ego.get_transform().get_matrix()));image_history.append(images)
            frame_history.append({'frame':frame,'seconds':snapshot.timestamp.elapsed_seconds})
            if tick%10==0:print(json.dumps({'stage':'tick','tick':tick,'frame':frame}),flush=True)
            if tick>=30 and (tick-30)%20==0:
                xyz,rot=local_right_handed_history(np.array(pose_history));index=len(records)
                path=args.out/f'context_{index:03d}.npz'
                np.savez_compressed(path,image_frames=np.stack(image_history,axis=1),ego_history_xyz=xyz,
                    ego_history_rot=rot,raw_carla_matrices=np.array(pose_history))
                actor_records=[]
                for actor in actors:
                    if actor.id==ego.id:continue
                    actor_records.append({'id':actor.id,'type':actor.type_id,'matrix':actor.get_transform().get_matrix(),
                        'extent':[actor.bounding_box.extent.x,actor.bounding_box.extent.y,actor.bounding_box.extent.z]})
                velocity=ego.get_velocity();record={'context':index,'path':str(path),'frames':list(frame_history),
                    'camera_frame_ids':[r['frame'] for r in list(frame_history)[-4:]],
                    'speed_mps':float(np.linalg.norm([velocity.x,velocity.y,velocity.z])),
                    'actors':actor_records,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                records.append(record);(args.out/'records.json').write_text(json.dumps(records,indent=2))
                print(json.dumps({'context':index,'frame':frame,'speed_mps':record['speed_mps']}),flush=True)
                if len(records)>=args.contexts:break
        manifest={'scope':__doc__,'carla_server':client.get_server_version(),'carla_client':carla.__version__ if hasattr(carla,'__version__') else None,
            'map':world.get_map().name,'port':args.port,'seed':args.seed,'target_speed':args.target_speed,'spawn_index':args.spawn_index,
            'controller':'Scripted waypoint lane follower, target 5m/s; NPCs if requested are static; no traffic manager',
            'dt':.1,'camera_order':[r[0] for r in RIG],'camera_indices':[0,1,2,6],
            'camera_resolution':[1920,1080],'camera_frames':4,'ego_history_poses':16,
            'coordinate_conversion':'M=diag(1,-1,1); p_right=M p_carla; R_right=M R_carla M; transform into newest ego pose',
            'contexts':len(records),'source_sha256':hashlib.sha256(source).hexdigest(),'wall_seconds':time.time()-start,
            'limitations':['Scripted lane-following contexts from one simulator map; no driving method evaluated','Contexts are not independent scenes','No closed-loop VLA or safety claim','CARLA visual domain differs from pretrained real-driving data']}
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    except BaseException:
        traceback.print_exc();raise
    finally:
        for sensor in sensors:
            try:sensor.stop();sensor.destroy()
            except RuntimeError:pass
        for actor in reversed(actors):
            try:actor.destroy()
            except RuntimeError:pass
        world.apply_settings(saved_settings)


if __name__=='__main__':main()
