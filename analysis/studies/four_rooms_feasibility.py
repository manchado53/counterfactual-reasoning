"""Exact reach probability for fixed known-map and random policies, not training."""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np


def check(room_scale):
    from training.envs.four_rooms import FourRoomsEnv
    env=FourRoomsEnv(room_scale)
    moves=np.asarray(env.moves).copy()
    distance=np.full(env.n_positions,100000,dtype=int)
    distance[env.goal]=0
    queue=deque([env.goal])
    while queue:
        node=queue.popleft()
        for neighbor in moves[node]:
            if distance[neighbor]>distance[node]+1:
                distance[neighbor]=distance[node]+1
                queue.append(neighbor)
    policy=distance[moves].argmin(axis=1)
    # Absorb goal mass to obtain the first-hit CDF at each step.
    moves[env.goal,:]=env.goal
    cases={}
    for name in ('known_map_shortest_path','uniform_random'):
        prob=(np.where(np.arange(4)[None,:]==policy[:,None],2/3,1/9)
              if name=='known_map_shortest_path' else np.full((env.n_positions,4),.25))
        mass=np.zeros(env.n_positions);mass[env.start]=1.
        weighted_length=0.;previous=0.
        for step in range(1,env.max_steps+1):
            mass=np.bincount(moves.ravel(),weights=(mass[:,None]*prob).ravel(),minlength=env.n_positions)
            reached=float(mass[env.goal])
            weighted_length+=step*(reached-previous)
            previous=reached
        cases[name]=dict(goal_probability=reached,
                         successful_length=weighted_length/reached if reached else None)
    return dict(room_scale=room_scale,open_cells=env.n_positions,cap=env.max_steps,
                shortest_steps=int(distance[env.start]),policies=cases,
                scope='Exact fixed-policy dynamics check; not a learned agent result')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    result=[check(1),check(2)]
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
