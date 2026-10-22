from collections import deque
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.envs.four_rooms import FourRoomsEnv, scaled_layout, LAYOUT, DOORS
from training.learners.four_rooms.build import config_for, validate_config, ARMS, smoke_overrides, build_agent
from analysis.studies.four_rooms_feasibility import check


def test_scale_one_preserves_published_map():
    assert scaled_layout(1)==(LAYOUT,DOORS)
    for bad in (0,3,2.,True):
        with pytest.raises(ValueError): scaled_layout(bad)


def test_double_interior_dimensions_keep_four_single_cell_doors():
    env=FourRoomsEnv(2)
    assert len(env.layout)==23 and all(len(row)==23 for row in env.layout)
    assert env.n_positions==404
    assert env.doors==((5,11),(11,3),(13,16),(18,11))
    assert env.cells[env.start]==(1,1) and env.cells[env.goal]==(16,16)
    cells=set(env.cells)-set(env.doors)
    sizes=[]
    while cells:
        queue=deque([cells.pop()]);size=0
        while queue:
            r,c=queue.popleft();size+=1
            for candidate in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if candidate in cells:
                    cells.remove(candidate);queue.append(candidate)
        sizes.append(size)
    assert sorted(sizes)==[80,100,100,120]
    for r,c in env.doors:
        # Exactly two neighbors: a one-cell passage between two rooms.
        assert sum(pos in env.index for pos in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))==2


def test_2x_wall_goal_timeout_and_probability():
    env=FourRoomsEnv(2);n=env.n_positions
    _,ns,r,done,_=env.transition(jnp.int32(env.start),0)
    assert int(ns)%n==env.start and not bool(done) and float(r)==0
    before=env.index[(16,15)]
    _,goal,r,done,info=env.transition(jnp.int32(before+199*n),3)
    assert bool(done) and bool(info['terminated']) and not bool(info['truncated']) and float(r)==1
    _,same,r,done,_=env.transition(goal,0)
    assert int(same)==int(goal) and float(r)==0 and bool(done)
    _,ns,r,done,info=env.transition(jnp.int32(env.start+199*n),0)
    assert bool(done) and bool(info['truncated']) and not bool(info['bootstrap'])
    assert env.get_obs(ns).shape==(405,)
    state=jnp.int32(env.index[(2,2)])
    keys=jax.random.split(jax.random.PRNGKey(71),36000)
    for action in range(4):
        ns=np.asarray(jax.jit(jax.vmap(lambda k:env.step(k,state,action)[1]))(keys))%n
        observed=np.array([(ns==int(env.moves[state,a])).mean() for a in range(4)])
        expected=np.full(4,1/9);expected[action]=2/3
        np.testing.assert_allclose(observed,expected,atol=.009)


def test_cap_feasibility_and_harder_random_exploration():
    old,new=check(1),check(2)
    assert new['shortest_steps']>old['shortest_steps']
    assert new['policies']['known_map_shortest_path']['goal_probability']>.99
    assert new['policies']['uniform_random']['goal_probability']<old['policies']['uniform_random']['goal_probability']


def test_frozen_recipe_and_only_size_changes():
    root=Path(__file__).parents[1]
    rows=json.loads((root/'experiments/four_rooms/cce-transfer-2x/training.json').read_text())['rows']
    assert len(rows)==15
    assert [r['arm'] for r in rows[:3]]==list(ARMS)
    for r in rows:
        c=r['config'];validate_config(c)
        old=config_for(r['arm'],r['seed'])
        delta={k for k in c.keys()|old.keys() if c.get(k)!=old.get(k)}
        assert delta=={'room_scale','protocol'}
    for seed in range(5):
        configs=[r['config'] for r in rows if r['seed']==seed]
        assert len({json.dumps({k:v for k,v in c.items() if k!='arm'},sort_keys=True) for c in configs})==1


def test_2x_bank_and_network_match_across_arms():
    identity=None;bank=None;weights=None
    for arm in ARMS:
        c=dict(config_for(arm,0,2),**smoke_overrides({}))
        agent=build_agent(c)
        assert agent.env.n_positions==404
        if identity is None:
            identity=agent.evaluation_identity();bank=agent.bank.copy()
            weights=[np.asarray(x) for x in jax.tree.leaves(agent.params)]
        assert agent.evaluation_identity()==identity
        np.testing.assert_array_equal(agent.bank,bank)
        for a,b in zip(weights,jax.tree.leaves(agent.params)):
            np.testing.assert_array_equal(a,b)
        del agent
