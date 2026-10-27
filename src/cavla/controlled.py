"""Small image-command decision problem with exact, explicitly defined effects.

This is a one-step synthetic decision task, not a validated driving simulator.
Commands choose a lane; a blocker in that lane limits forward progress. A matched
intervention moves the blocker closer or changes an irrelevant billboard. Each
scene yields four inputs: (world0,left), (world1,left), (world0,right), (world1,right).
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class State:
    blocker_lane: int
    distance: float
    background: float
    billboard: float


COMMANDS = ("take the left lane", "take the right lane")


def oracle(state, command):
    lane = -1 if command == 0 else 1
    forward = min(.85, max(0., state.distance-.12)) if lane == state.blocker_lane else .85
    return np.array([lane*.65, forward], dtype=np.float32)


def render(state, size=24):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    im = np.full((3, size, size), state.background, dtype=np.float32)
    im[:, :, size//6:5*size//6] = .14
    im[:, :, size//2] = .36
    center_x = size*(.30 if state.blocker_lane == -1 else .70)
    center_y = (1-state.distance)*(.72*size)+.08*size
    blob = np.exp(-((xx-center_x)**2+(yy-center_y)**2)/(2*1.0**2))
    im[0] = np.clip(im[0]+.8*blob, 0, 1)
    im[1] = np.clip(im[1]-.1*blob, 0, 1)
    im[2, :3, :3] = state.billboard
    # Ego marker is identical for all worlds and commands.
    im[1, -3:-1, size//2-1:size//2+1] = .9
    return im


def world_effect(values):
    """Two world effects, one per command. Supports numpy arrays and tensors."""
    return values[:, [1, 3]]-values[:, [0, 2]]


def mixed_effect(values):
    effects = world_effect(values)
    return effects[:, 1]-effects[:, 0]


def make_dataset(n_scenes, seed, shift="iid"):
    rng = np.random.default_rng(seed)
    images, targets, nulls, latent_states = [], [], [], []
    for _ in range(n_scenes):
        lane = int(rng.choice([-1, 1]))
        distance = float(rng.uniform(.60, .90))
        nuisance = bool(rng.random() < .25)
        severity = float(rng.uniform(.24, .40) if shift == "severity" else rng.uniform(.04, .18))
        # Correlation is known and documented, not a hidden data selection step.
        background = (.08 if lane == -1 else .30)
        if shift == "background":
            background = .38-background
        background += float(rng.uniform(-.025, .025))
        base = State(lane, distance, background, float(rng.uniform(.1, .9)))
        changed = (replace(base, billboard=1-base.billboard) if nuisance else
                   replace(base, distance=distance-severity))
        images.append(np.stack([render(base), render(changed), render(base), render(changed)]))
        targets.append(np.stack([oracle(base, 0), oracle(changed, 0), oracle(base, 1), oracle(changed, 1)]))
        nulls.append(nuisance)
        latent_states.append([lane, distance, severity, background, base.billboard])
    return {"images": np.stack(images), "commands": np.tile([0, 0, 1, 1], (n_scenes, 1)),
            "targets": np.stack(targets), "null": np.array(nulls),
            "states": np.array(latent_states),
            "ids": np.array([f"{seed}:{i}" for i in range(n_scenes)])}
