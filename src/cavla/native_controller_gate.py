"""Offline downstream-controller diagnostic with a common declared PID state.

No closed-loop world is run. Simulated prior steering values are not observed
controller histories. This gate tests whether trajectory error reaches controls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    import carla
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    root=Path('/mnt/datasets/alpamayo/carlamayo-integration');sys.path.insert(0,str(root))
    sys.path.append('/home/perceptron-lab/CARLA_0.9.16/PythonAPI/carla')
    from module.pid_controller import OfficialPIDFollower
    contexts=json.loads(Path('artifacts/carla_native_contexts_v3_20261022/records.json').read_text())
    arrays={}
    for folder in ['native_solver_probe_v2_20261022','native_solver_classical_controls_v1_20261022']:
        with np.load(Path('artifacts')/folder/'predictions.npz') as z:arrays.update({key:z[key] for key in z.files if z[key].shape==(64,3)})
    with np.load('artifacts/native_solver_convergence_bf16_v1_20261022/predictions.npz') as z:
        for context in contexts:
            key=f"context_{context['context']:03d}_s9202_heun160";arrays[key]=z[key]
    class Vehicle:
        def __init__(self,speed,steer):self.speed=speed;self.steer=steer
        def get_world(self):return None
        def get_transform(self):return carla.Transform()
        def get_velocity(self):return carla.Vector3D(x=self.speed)
        def get_control(self):return carla.VehicleControl(steer=self.steer)
    records=[]
    for context in contexts:
        prefix=f"context_{context['context']:03d}_s9202_";speed=context['speed_mps']
        for previous_steer in [-.4,0.,.4]:
            outputs={}
            for key,pred in arrays.items():
                if not key.startswith(prefix):continue
                vehicle=Vehicle(speed,previous_steer);controller=OfficialPIDFollower(None,vehicle)
                steer,throttle,brake,debug=controller.compute_control(vehicle.get_transform(),pred,speed)
                outputs[key[len(prefix):]]={'steer':steer,'throttle':throttle,'brake':brake,'debug':debug}
            ref=outputs['heun160']
            for solver,out in outputs.items():
                records.append({'context':context['context'],'seed':9202,'previous_steer':previous_steer,'solver':solver,
                    'steer_error':out['steer']-ref['steer'],'throttle_error':out['throttle']-ref['throttle'],
                    'brake_error':out['brake']-ref['brake'],'reference':ref,'actual':out})
    result={'scope':__doc__,'records':records,'limitations':['Selected seed and two windows from one route',
        'Common empty PID error buffers and three hypothetical previous steering values, not recorded controller memory',
        'Reference trajectory is numerical, not normative','No vehicle execution or safety assessment']}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    files=[root/'module/pid_controller.py',root/'module/config.py',Path('/home/perceptron-lab/CARLA_0.9.16/PythonAPI/carla/agents/navigation/controller.py')]
    (args.out/'manifest.json').write_text(json.dumps({'external_sources_read_only_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}},indent=2))


if __name__=='__main__':main()
