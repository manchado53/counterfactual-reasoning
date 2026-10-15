import copy
import json
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from training.envs import connect_four_minimax as task
from training.learners.connect_four.minimax_eval import root_scores
from training.learners.connect_four.protocol_v4 import ARMS, config_for, validate, TRAIN_BANK
from training.learners.connect_four.stochastic_cce import consequence, build_rollouts
from training.learners.connect_four.bank_v3 import load
from training.learners.connect_four.evaluation_v4 import Evaluator
from counterfactual_rl.replay.decay import DecayBuffer, blend_weight


class ZeroNetwork:
    def apply(self,params,obs):
        return jnp.zeros(obs.shape[:-1]+(7,),jnp.float32)


def boards():
    rng=np.random.default_rng(134)
    states=[]
    init=jax.jit(task.ENV.init)
    step=jax.jit(task.ENV.step)
    for game in range(12):
        s=init(jax.random.PRNGKey(game))
        for ply in range(30):
            if bool(s.terminated):break
            if ply%3==0:states.append(s)
            a=int(rng.choice(np.flatnonzero(np.asarray(s.legal_action_mask))))
            s=step(s,jnp.int32(a))
    return states


@pytest.fixture(scope='module')
def oracle_boards():
    return boards()


def test_gpu_search_matches_independent_cpu_minimax(oracle_boards):
    states=jax.tree.map(lambda *x:jnp.stack(x),*oracle_boards)
    actual=np.asarray(jax.jit(jax.vmap(task.search_scores))(states))
    for s,row in zip(oracle_boards,actual):
        obs=np.asarray(s.observation,dtype=np.int8)
        b=(obs[:,:,0]-obs[:,:,1]).reshape(42).copy()
        np.testing.assert_array_equal(row,root_scores(b,3))


def test_foe_ties_are_seeded_legal_and_variable(oracle_boards):
    search=jax.jit(task.search_scores)
    for s in oracle_boards:
        scores=np.asarray(search(s));best=np.flatnonzero(scores==scores.max())
        if len(best)>1:break
    else:raise AssertionError('No tied fixture')
    keys=jax.random.split(jax.random.PRNGKey(15),128)
    f=jax.jit(jax.vmap(task.opponent,in_axes=(None,0)))
    a=np.asarray(f(s,keys));b=np.asarray(f(s,keys))
    np.testing.assert_array_equal(a,b)
    assert set(a)<=set(best) and len(set(a))>1
    assert np.asarray(s.legal_action_mask)[a].all()


def test_terminal_reward_and_absorption():
    f=jax.jit(task.step)
    for seed in (0,1,2):
        s=task.ENV.init(jax.random.PRNGKey(seed));learner=s.current_player
        for a in [0,1,0,1,0,2]:s=task.ENV.step(s,a)
        r=f(s,learner,jnp.int32(0),jax.random.PRNGKey(5))
        assert bool(r.done) and float(r.reward)==1.
        again=f(r.state,learner,jnp.int32(0),jax.random.PRNGKey(7))
        assert float(again.reward)==0. and int(again.board_moves)==0


def test_stochastic_collector_preserves_split_carry():
    collect=task.build_collect(ZeroNetwork())
    initial=jax.jit(task.initial_carry)(jax.random.split(jax.random.PRNGKey(30),4))
    keys=jax.random.split(jax.random.PRNGKey(4),48*4).reshape(48,4,2)
    whole,rows=collect(None,jnp.float32(.5),initial,keys)
    half,a=collect(None,jnp.float32(.5),initial,keys[:17])
    split,b=collect(None,jnp.float32(.5),half,keys[17:])
    for x,y in zip(jax.tree.leaves(whole),jax.tree.leaves(split)):np.testing.assert_array_equal(x,y)
    joined=jax.tree.map(lambda x,y:jnp.concatenate([x,y]),a,b)
    for x,y in zip(jax.tree.leaves(rows),jax.tree.leaves(joined)):np.testing.assert_array_equal(x,y)
    assert np.take_along_axis(np.asarray(rows['mask']),np.asarray(rows['a'])[...,None],-1).all()
    assert np.asarray(rows['done']).any()
    assert set(np.asarray(rows['role']).ravel())=={0,1}


def test_empirical_wasserstein_sorted_samples_and_legal_weights():
    returns=np.zeros((3,7,4))
    returns[0,0]=[-1,1,-1,1];returns[0,1]=[1,-1,1,-1]
    returns[1,0]=[-1,-1,-1,-1];returns[1,1]=[1,1,1,1]
    returns[:,2:]=999
    masks=jnp.array([[1,1,0,0,0,0,0],[1,1,0,0,0,0,0],[1,0,0,0,0,0,0]],bool)
    scores=consequence(returns,jnp.zeros((3,7),jnp.float32),masks,np.zeros(3,int))
    np.testing.assert_allclose(scores,[0,2,0])


def test_rollout_four_paths_identity_and_no_global_rng():
    network=ZeroNetwork();f=build_rollouts(network)
    states,players=jax.vmap(task.reset)(jax.random.split(jax.random.PRNGKey(4),2),jnp.array([0,1],jnp.int32))
    keys=jax.random.split(jax.random.PRNGKey(7),8).reshape(2,4,2)
    before=pickle.dumps(np.random.get_state())
    a,done=f(None,None,states,players,jnp.arange(7,dtype=jnp.int32),keys)
    b,done2=f(None,None,states,players,jnp.arange(7,dtype=jnp.int32),keys)
    assert a.shape==(2,7,4) and np.asarray(done).all() and np.asarray(done2).all()
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all() and pickle.dumps(np.random.get_state())==before


def test_all_configs_train_minimax3_and_match():
    from training.learners.connect_four.build import validate_config,smoke_overrides
    rows=json.loads(Path('experiments/connect_four/minimax3-v4/training.json').read_text())['rows']
    assert len(rows)==15
    assert {(r['arm'],r['seed']) for r in rows}=={(a,s) for a in ARMS for s in range(5)}
    replay={'arm','score_on_insert','decay_enabled','normalise_priorities'}
    for seed in range(5):
        configs=[config_for(a,seed) for a in ARMS]
        assert all(c['training_opponent']==TRAIN_BANK and c['training_opponent']['depth']==3 for c in configs)
        assert all(c['cf_n_rollouts']==4 and c['primary_eval_depth']==3 for c in configs)
        comparable=[{k:v for k,v in c.items() if k not in replay} for c in configs]
        assert comparable[0]==comparable[1]==comparable[2]
    for r in rows:
        validate_config(r['config']);validate(dict(r['config'],**smoke_overrides(r['config'])))
    bad=copy.deepcopy(rows[0]['config']);bad['training_opponent']['depth']=2
    with pytest.raises(ValueError):validate(bad)


def test_primary_depth3_fixed_bank_identity_and_rng():
    ev=Evaluator(ZeroNetwork(),load()[1])
    before=pickle.dumps(np.random.get_state())
    first=ev.evaluate(None);paths=ev.last_actions.copy()
    second=ev.evaluate(None)
    assert first==second and pickle.dumps(np.random.get_state())==before
    np.testing.assert_array_equal(paths,ev.last_actions)
    assert first['primary_depth']==3 and first['primary']['cases']==200
    assert first['primary_by_suite']['empty']['cases']==100
    assert first['primary_by_suite']['opening']['cases']==100
    expected=sum(first['by_suite_role_depth'][f'{s}:{r}:3']['wins'] for s in ('empty','opening') for r in (0,1))
    assert first['primary']['wins']==expected


def test_score_before_draw_and_finite_update():
    from training.learners.connect_four.build import build_agent
    agent=build_agent(config_for('cce_decay',0))
    cfg=agent.config.copy()
    agent.config.update(n_envs=2,collect_steps=2,batch_size=4)
    agent.carry=jax.jit(task.initial_carry)(jax.random.split(jax.random.PRNGKey(50),2))
    try:
        agent.collect()
        assert (agent.buffer.scored_at[:len(agent.buffer)]>=0).all()
        batch,indices,weights=agent.sample()
        params,_,td,diag=agent.update_fn(agent.params,agent.target_params,agent.opt_state,batch,weights)
        assert np.isfinite(td).all() and np.isfinite(diag).all()
        assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(params),jax.tree.leaves(agent.params)))
        agent.buffer.scored_at[0]=-1
        with pytest.raises(AssertionError,match='Unscored'):agent.sample()
    finally:agent.config=cfg


@pytest.mark.parametrize('episode,weight',[(0,0),(100000,0),(125000,.5),(150000,1),(200000,1)])
def test_decay_stays_at_source_boundaries(episode,weight):
    assert blend_weight(episode)==weight
