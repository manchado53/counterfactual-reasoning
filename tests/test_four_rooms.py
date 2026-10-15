"""Task, shared replay and controlled-study contracts."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.envs.four_rooms import FourRoomsEnv, DOORS, LAYOUT
from training.learners.four_rooms.build import ARMS, config_for, smoke_overrides, build_agent, validate_config
from training.learners.shared.discrete import build_branches
from counterfactual_rl.replay.rank_buffer import SelectionReplayBuffer
from counterfactual_rl.scoring.weighted_wasserstein import weighted_wasserstein_scores


def test_map_and_exact_transition_graph():
    env = FourRoomsEnv()
    assert len(env.cells) == 104
    assert all(LAYOUT[r][c] != '#' for r, c in DOORS)
    assert sum(row.count('#') for row in LAYOUT) == 65
    reachable = {env.start}
    for _ in range(104):
        reachable |= {int(env.moves[s, a]) for s in list(reachable) for a in range(4)}
    assert reachable == set(range(104))
    for s, (r, c) in enumerate(env.cells):
        for a, (dr, dc) in enumerate(((-1,0),(1,0),(0,-1),(0,1))):
            assert int(env.moves[s,a]) == env.index.get((r+dr,c+dc), s)


def test_wall_collision():
    env = FourRoomsEnv()
    for action in (0, 2):
        _, ns, r, done, _ = env.transition(jnp.int32(env.start), action)
        assert int(ns) % 104 == env.start
        assert int(ns) // 104 == 1
        assert float(r) == 0 and not bool(done)


def test_movement_probability_and_action_order():
    env = FourRoomsEnv()
    s = jnp.int32(env.index[(2, 2)])
    keys = jax.random.split(jax.random.PRNGKey(17), 90000)
    for action in range(4):
        ns = np.asarray(jax.jit(jax.vmap(lambda k: env.step(k, s, action)[1]))(keys)) % 104
        expected = np.full(4, 1/9); expected[action] = 2/3
        actual = np.array([(ns == int(env.moves[s, a])).mean() for a in range(4)])
        np.testing.assert_allclose(actual, expected, atol=.006)


def test_goal_timeout_and_absorbing_reward():
    env = FourRoomsEnv()
    before = jnp.int32(env.index[(9,8)])
    for elapsed in (0, 199):
        _, goal, r, done, info = env.transition(before + elapsed*104, 3)
        assert float(r) == 1 and bool(done) and bool(info['terminated'])
        assert not bool(info['truncated'])
        _, same, r, done, _ = env.transition(goal, 0)
        assert int(same) == int(goal) and float(r) == 0 and bool(done)
    _, ns, r, done, info = env.transition(jnp.int32(env.start + 199*104), 0)
    assert bool(done) and bool(info['truncated']) and not bool(info['bootstrap'])
    assert float(r) == 0 and int(ns) // 104 == 200
    assert env.get_obs(jnp.int32(env.start)).shape == (105,)
    assert float(env.get_obs(ns)[-1]) == 1.


def test_recipe_parity():
    recipe = json.loads((Path(__file__).parents[1]/'experiments/four_rooms/cce-transfer/training.json').read_text())
    assert len(recipe['rows']) == 15
    for seed in range(5):
        configs = [r['config'] for r in recipe['rows'] if r['seed'] == seed]
        for c in configs:
            validate_config(c)
        assert len({json.dumps({k:v for k,v in c.items() if k!='arm'},sort_keys=True) for c in configs}) == 1
    bad = config_for('cce',0); bad['alpha'] *= 2
    with pytest.raises(ValueError): validate_config(bad)


def test_rank_per_is_inverse_rank_and_ties_equal():
    b = SelectionReplayBuffer(capacity=4, selection='rank_per', rank_alpha=.25)
    b.add_batch(dict(s=np.arange(4), a=np.zeros(4), r=np.zeros(4), done=np.zeros(4)))
    b.update_priorities(np.arange(4), np.array([8., 4., 4., 1.]))
    p = np.array([1.,2.5,2.5,4.])**-.25; p /= p.sum()
    np.testing.assert_allclose(b._compute_priorities(), p)
    b.selection='uniform'; b._cached_probs=None
    np.testing.assert_allclose(b._compute_priorities(), [.25]*4)


def test_branch_sampler_matches_manual_exact_simulator():
    env = FourRoomsEnv()
    policy = jnp.zeros(env.n_states,dtype=jnp.int32)
    values = jnp.full(env.n_states, 2.)
    states = jnp.array([env.start, env.start+199*104, env.goal])
    keys = jax.random.split(jax.random.PRNGKey(8),12).reshape(3,4,2)
    actual = np.asarray(build_branches(env,3,.99)(policy,values,states,jnp.arange(4),keys))
    for i,s in enumerate(states):
        for a in range(4):
            state,key,total,discount,ended = s,keys[i,a],0.,1.,False
            for t in range(3):
                key,sk = jax.random.split(key)
                _,state,r,done,_=env.step(sk,state,a if t==0 else 0)
                if not ended: total += discount*float(r)
                discount *= .99; ended |= bool(done)
            if not ended: total += discount*2
            assert actual[i,a] == pytest.approx(total,abs=1e-6)


def test_fixed_bank_rng_isolation_and_shared_update():
    agents = [build_agent(dict(config_for(a,0), **smoke_overrides({}))) for a in ARMS]
    ref = agents[0]
    for agent in agents:
        assert agent.evaluation_identity() == ref.evaluation_identity()
        np.testing.assert_array_equal(agent.bank,ref.bank)
        before = np.asarray(agent.key).copy()
        np.random.seed(99); rng_before = np.random.get_state()
        assert agent.evaluate() == ref.evaluate()
        np.testing.assert_array_equal(agent.key,before)
        assert np.random.get_state()[2:] == rng_before[2:]
        np.testing.assert_array_equal(np.random.get_state()[1],rng_before[1])
        for a,b in zip(jax.tree.leaves(agent.params),jax.tree.leaves(ref.params)):
            np.testing.assert_array_equal(a,b)
    keys=jax.random.split(jax.random.PRNGKey(9),128).reshape(4,32,2)
    outputs=[]
    for agent in agents:
        _,out=agent.collect_fn(agent.params,jnp.float32(1.),agent.states,keys)
        outputs.append(out); agent.insert(out)
    for out in outputs[1:]:
        for a,b in zip(out,outputs[0]): np.testing.assert_array_equal(a,b)
    cce=agents[-1]
    assert cce.scored == 128 and (cce.buffer.scored_at[:128]>=0).all()
    # Taken actions are retained exactly, not replaced with greedy actions.
    np.testing.assert_array_equal([t['a'] for t in cce.buffer.buffer[:128]],np.asarray(outputs[-1][1]).ravel())
    # The same supplied minibatch and weights give bit-identical updates in every arm.
    s,a,r,ns,d=[np.asarray(x).ravel()[:8] for x in outputs[0]]
    updated=[]
    for agent in agents:
        updated.append(agent.update_fn(agent.params,agent.target,agent.opt_state,
            agent.env.get_obs(jnp.asarray(s)),jnp.asarray(a),jnp.asarray(r),
            agent.env.get_obs(jnp.asarray(ns)),jnp.asarray(d),jnp.ones(8)))
    for result in updated[1:]:
        for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(updated[0])):
            np.testing.assert_array_equal(a,b)


def test_score_excludes_taken_action():
    returns=np.array([[0.,1.,2.,4.]]*4)
    actions=np.arange(4)
    scores=weighted_wasserstein_scores(returns,np.full((4,4),.25),actions)
    np.testing.assert_allclose(scores,[7/3,5/3,5/3,3.])
