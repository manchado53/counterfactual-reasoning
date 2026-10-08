"""Task-specific fixed-recipe transfer checks; CPU-safe, no training submissions."""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from counterfactual_rl.agents.doorkey.adapter import (
    Carry, DoorKeyAdapter, initial_carry, ddqn_targets, weighted_scores)
from counterfactual_rl.agents.doorkey.train import Trainer, ARMS, config_for, source_hashes


def same(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_split_random_collection_and_full_timeout():
    task = DoorKeyAdapter()
    params = task.net.init(jax.random.PRNGKey(0), jnp.int32(0))
    keys = jax.random.split(jax.random.PRNGKey(123), 4*128).reshape(4,128,2)
    end, whole = task.collect(params, jnp.float32(1), initial_carry(4), keys)
    carry = initial_carry(4)
    pieces = []
    for start in range(0,128,32):
        carry, part = task.collect(params, jnp.float32(1), carry, keys[:,start:start+32])
        pieces.append(part)
    same(carry, end)
    same(jax.tree.map(lambda *xs: jnp.concatenate(xs,axis=1), *pieces), whole)
    assert not np.asarray(whole['terminal_start']).any()
    assert (np.asarray(whole['episode_length'])[np.asarray(whole['truncated'])] == 50).all()
    assert np.asarray(whole['truncated']).any()
    assert not np.asarray(whole['done'])[np.asarray(whole['truncated'])].any()


@pytest.mark.parametrize('goal_step', [31,32,33])
def test_goal_around_chunk_edge_once(goal_step):
    task = DoorKeyAdapter()
    # Stand next to goal, facing south, key held, door open.
    s = task.env._index[(3,4,1,1,2)]
    carry = initial_carry(1)._replace(state=jnp.array([s]), obs=jnp.array([s]))
    actions = jnp.full((1,64), 6).at[0,goal_step-1].set(2)
    end, whole = task.collect_actions(carry, actions)
    mid, first = task.collect_actions(carry, actions[:,:32])
    split_end, second = task.collect_actions(mid, actions[:,32:])
    same(end, split_end)
    same(whole, jax.tree.map(lambda a,b: jnp.concatenate([a,b],axis=1),first,second))
    assert np.asarray(whole['r']).sum() == 1
    assert np.asarray(whole['done']).sum() == 1
    assert not np.asarray(whole['terminal_start']).any()
    assert int(whole['s'][0,goal_step]) == 0
    assert int(whole['episode_length'][0,goal_step-1]) == goal_step


@pytest.mark.parametrize('split', [49,50,51])
def test_timeout_carry_and_real_next_state(split):
    task = DoorKeyAdapter()
    # Left turns stay live; 50 actions end collection only.
    actions = jnp.zeros((1,100), dtype=jnp.int32)
    end, rows = task.collect_actions(initial_carry(1), actions)
    c, x = task.collect_actions(initial_carry(1), actions[:,:split])
    c, y = task.collect_actions(c, actions[:,split:])
    same(end,c)
    same(rows,jax.tree.map(lambda a,b:jnp.concatenate([a,b],axis=1),x,y))
    assert not np.asarray(rows['done']).any()
    assert list(np.flatnonzero(rows['truncated'][0])) == [49,99]
    assert int(rows['s'][0,50]) == 0
    assert int(rows['next_s'][0,49]) != 0  # real west-facing state, not reset east
    assert list(np.asarray(rows['episode_length'])[np.asarray(rows['ended'])]) == [50,50]


def test_double_dqn_terminal_vs_truncation():
    online = jnp.array([[5.,1.],[5.,1.]])
    target = jnp.array([[2.,9.],[2.,9.]])
    result = ddqn_targets(online,target,jnp.array([0.,1.]),jnp.array([False,True]))
    np.testing.assert_allclose(result,[1.98,1.])


def constant_params(task, value):
    p = task.net.init(jax.random.PRNGKey(2),jnp.int32(0))
    p = jax.tree.map(jnp.zeros_like,p)
    p['params']['Dense_3']['bias'] = jnp.full((7,),value,dtype=jnp.float32)
    return p


def test_cce_tail_discount_and_terminal_mask():
    task = DoorKeyAdapter()
    p = constant_params(task,0.)
    tp = constant_params(task,2.)
    goal_front = task.env._index[(3,4,1,1,2)]
    returns = np.asarray(task.branch_returns(p,tp,jnp.array([0,goal_front])))
    np.testing.assert_allclose(returns[0], 2*.99**60, rtol=2e-6)
    assert returns[1,2] == 1.  # first-action goal; never add target tail


def test_cce_can_reach_reward_after_collection_cutoff():
    task = DoorKeyAdapter()
    n = task.env.n_states
    task.next_state = jnp.broadcast_to(jnp.minimum(jnp.arange(n)+1,55)[:,None],(n,7))
    task.reward = jnp.broadcast_to((jnp.arange(n)==54)[:,None],(n,7)).astype(jnp.float32)
    task.terminal = jnp.broadcast_to((jnp.arange(n)>=54)[:,None],(n,7))
    p = constant_params(task,0.)
    result = task.branch_returns(p,p,jnp.array([0]))
    np.testing.assert_allclose(result,.99**54,rtol=2e-6)


@pytest.mark.parametrize('seed', [3,17,91])
def test_table_rollout_matches_stepwise_network(seed):
    task = DoorKeyAdapter()
    params = task.net.init(jax.random.PRNGKey(seed),jnp.int32(0))
    target = task.net.init(jax.random.PRNGKey(seed+100),jnp.int32(0))
    rng = np.random.default_rng(seed)
    ids = rng.choice(task.env.non_terminal, size=16, replace=False)
    ids[-1] = task.env._index[(3,4,1,1,2)]
    states = jnp.asarray(ids)

    def reference(s, first):
        key = jax.random.PRNGKey(0)
        _, ns, reward, done, _ = task.env.step(key,s,first)
        def step(c, _):
            s, ret, disc, ended = c
            a = jnp.argmax(task.net.apply(params,s))
            _, ns, reward, done, _ = task.env.step(key,s,a)
            return (ns, ret+jnp.where(ended,0.,disc*reward),
                    jnp.where(ended,disc,disc*.99), ended|done), None
        (s,ret,disc,ended),_ = jax.lax.scan(step,
            (ns,reward,jnp.float32(.99),done),None,length=59)
        return ret+jnp.where(ended,0.,disc*jnp.max(task.net.apply(target,s)))

    ref = jax.jit(jax.vmap(jax.vmap(reference,in_axes=(None,0)),in_axes=(0,None)))
    actual = np.asarray(task.branch_returns(params,target,states))
    expected = np.asarray(ref(states,jnp.arange(7)))
    # Batched GEMM can round differently. Compare action choices too, so a small
    # return tolerance cannot conceal a changed continuation path on these seeds.
    all_ids = jnp.arange(task.env.n_states)
    batched = task.net.apply(params,all_ids)
    scalar = jax.vmap(lambda s:task.net.apply(params,s))(all_ids)
    np.testing.assert_array_equal(np.argmax(batched,axis=-1),np.argmax(scalar,axis=-1))
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)


def test_scores_match_independent_wasserstein_reference():
    rng = np.random.default_rng(42)
    r = rng.normal(size=(12,7)).astype(np.float32)
    p = rng.uniform(size=(12,7)).astype(np.float32)
    taken = np.arange(12)%7
    expected = []
    for i,a in enumerate(taken):
        alternatives = [b for b in range(7) if b != a]
        expected.append(sum(float(p[i,b])*wasserstein_distance([r[i,a]],[r[i,b]]) for b in alternatives)
                        /sum(float(p[i,b]) for b in alternatives))
    np.testing.assert_array_equal(weighted_scores(r,p,taken),expected)
    np.testing.assert_allclose(weighted_scores(r,np.zeros_like(p),taken),
        [np.mean([abs(float(r[i,a])-float(r[i,b])) for b in range(7) if b != a]) for i,a in enumerate(taken)])


@pytest.mark.parametrize('arm', ARMS)
def test_actual_update_recipe_eval_and_checkpoint(arm,tmp_path):
    np.random.seed(0)
    agent = Trainer(arm,0)
    cfg = config_for(arm,0)
    assert (cfg['hidden_dim'],cfg['n_layers'],cfg['batch_size'],cfg['collect_steps']) == (128,3,128,32)
    assert cfg['alpha'] == .0003 and cfg['double_dqn'] is True
    before = agent.rng_state()
    initial_eval = agent.evaluate()
    assert agent.rng_state() == before
    batch = jnp.zeros(128,dtype=jnp.int32)
    prior = jax.tree.map(np.array,agent.params)
    agent.params,agent.opt_state,td,stats = agent.update_fn(agent.params,agent.target,agent.opt_state,
        batch,batch,jnp.ones(128),batch,jnp.ones(128,dtype=bool),jnp.ones(128))
    assert np.isfinite(np.asarray(stats)).all()
    assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(prior),jax.tree.leaves(agent.params)))
    assert all(x.shape == (128,) for x in [td])
    before = agent.rng_state()
    path = tmp_path/'check.pkl'
    agent.save(path)
    check = Trainer(arm,0)
    check.load(path)
    same(agent.params,check.params)
    same(agent.target,check.target)
    same(agent.opt_state,check.opt_state)
    assert check.evaluate() == agent.evaluate()
    assert agent.rng_state() == before
    assert initial_eval['eval_episodes'] == 1000 and initial_eval['unique_tasks'] == 1
    assert source_hashes()


@pytest.mark.parametrize('arm', ARMS)
def test_source_sized_collection_insert_and_actual_training(arm):
    np.random.seed(0)
    agent = Trainer(arm,0)
    row = agent.train_block()
    assert row['env_steps'] == 8192
    assert row['updates'] == row['update_attempts'] == 512
    assert len(agent.buffer) == 8192
    assert row['terminal_start_transitions'] == 0
    assert np.isfinite(row['loss'])
    if agent.cce:
        assert np.all(agent.buffer.scored_at[:8192] == 0)
        assert np.isfinite(agent.buffer.consequence_scores[:8192]).all()
        assert row['scored_frac'] == 1.
    before = agent.rng_state()
    diagnostic = agent.diagnostics()
    assert agent.rng_state() == before
    assert diagnostic['n'] == 8192
