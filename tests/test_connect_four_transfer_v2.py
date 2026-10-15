import copy
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from training.envs import connect_four_transfer as task
from training.learners.connect_four.bank_v2 import load, verify, canonical
from training.learners.connect_four.protocol_v2 import config_for, ARMS, validate
from training.learners.connect_four.transfer import consequence, targets
from training.learners.connect_four.transfer_v2 import Agent, Network, build_update
from training.learners.connect_four.evaluation_v2 import foe
from counterfactual_rl.replay.decay import DecayBuffer, blend_weight


def test_bank_legal_unique_balanced_and_hash(tmp_path):
    bank, digest = load()
    assert verify(bank) == digest
    assert len({tuple(np.array(c['board']).ravel()) for c in bank['cases']}) == 1000
    assert len({len(c['opening']) for c in bank['cases']}) > 4
    p = tmp_path/'bank.json'
    p.write_bytes(canonical(bank))
    assert load(digest, p)[1] == digest
    bank['cases'][0]['seed'] += 1
    p.write_bytes(canonical(bank))
    with pytest.raises(ValueError, match='hash'):
        load(digest, p)


def test_matched_configs_and_recipe():
    replay = {'arm','score_on_insert','decay_enabled','normalise_priorities'}
    for seed in range(5):
        configs = [config_for(a, seed) for a in ARMS]
        for cfg in configs:
            validate(cfg)
        controls = [{k:v for k,v in c.items() if k not in replay} for c in configs]
        assert controls[0] == controls[1] == controls[2]
    p = Path(__file__).resolve().parents[1]/'experiments/connect_four/jaxnav-transfer-v2/training.json'
    rows = json.loads(p.read_text())['rows']
    assert len(rows) == 15
    assert {(r['arm'],r['seed']) for r in rows} == {(a,s) for a in ARMS for s in range(5)}
    for r in rows:
        validate(r['config'])
    bad = config_for('per', 0)
    bad['alpha'] = .0003
    with pytest.raises(ValueError):
        validate(bad)


@pytest.mark.parametrize('episode,weight', [(0,0),(99999,0),(100000,0),(125000,.5),(150000,1),(200000,1)])
def test_decay_boundaries(episode, weight):
    assert blend_weight(episode) == weight


def test_decay_probabilities_cache_eviction_and_weights():
    b = DecayBuffer(capacity=4, normalise_priorities='rank', priority_mixing='multiplicative')
    for i in range(4):
        b.add({'r':0})
    b.update_consequence_scores(np.arange(4), np.array([0,0,1,4]), 0)
    b.update_priorities(np.arange(4), np.array([4,1,1,0]))
    p0 = b._compute_priorities().copy()
    td = b.ranked_td()
    b.blend = .5
    np.testing.assert_allclose(b._compute_priorities(), (p0+td)/2)
    b.blend = 1
    np.testing.assert_array_equal(b._compute_priorities(), td)
    np.random.seed(9)
    _, indices, weights = b.sample(4)
    np.testing.assert_allclose(weights, 1/(4*td[indices]))
    b.update_priorities(np.array([0]), np.array([0.]))
    assert not np.array_equal(b._compute_priorities(), td)
    b.add({'r':1})
    assert b.scored_at[0] == -1
    b.update_consequence_scores(np.array([0]), np.array([5.]), 3)
    np.testing.assert_allclose(b._compute_priorities(), b.ranked_td())
    assert b._compute_priorities().sum() == pytest.approx(1)


def test_cce_legal_alternatives_and_known_return_difference():
    q = jnp.zeros((2,7))
    returns = jnp.array([[1.,-1.,900,900,900,900,900],[1.,900,900,900,900,900,900]])
    mask = jnp.array([[1,1,0,0,0,0,0],[1,0,0,0,0,0,0]], bool)
    np.testing.assert_allclose(consequence(returns,q,mask,jnp.array([0,0])), [2.,0.])


def test_targets_ignore_illegal_actions_and_terminal_infinity():
    online = jnp.array([[1.,999.,0.],[1.,999.,0.]])
    target = jnp.array([[3.,999.,2.],[jnp.inf,jnp.inf,jnp.inf]])
    mask = jnp.array([[1,0,1],[0,0,0]], bool)
    np.testing.assert_allclose(targets(jnp.array([0.,1.]),jnp.array([False,True]),online,target,mask,.999),[2.997,1.])


class ZeroNetwork:
    def apply(self, params, obs):
        return jnp.zeros(obs.shape[:-1]+(7,))


def test_collector_split_carry_and_legal_actions():
    collect = task.build_collect(ZeroNetwork())
    initial = task.initial_carry(jax.random.split(jax.random.PRNGKey(30), 4))
    keys = jax.random.split(jax.random.PRNGKey(4), 4*48).reshape(48,4,2)
    whole, rows = collect(None, .5, initial, keys)
    half, a = collect(None, .5, initial, keys[:17])
    split, b = collect(None, .5, half, keys[17:])
    for x,y in zip(jax.tree.leaves(whole),jax.tree.leaves(split)):
        np.testing.assert_array_equal(x,y)
    joined = jax.tree.map(lambda x,y: jnp.concatenate([x,y]), a,b)
    for x,y in zip(jax.tree.leaves(rows),jax.tree.leaves(joined)):
        np.testing.assert_array_equal(x,y)
    mask, actions = np.asarray(rows['mask']), np.asarray(rows['a'])
    assert np.take_along_axis(mask,actions[...,None],-1).all()
    assert np.asarray(rows['done']).any()
    assert np.max(rows['episode_length']) > 1


def test_reward_sign_both_pgx_player_ids():
    step = jax.jit(task.step)
    seen = set()
    for seed in range(8):
        s = task.ENV.init(jax.random.PRNGKey(seed))
        learner = int(s.current_player)
        seen.add(learner)
        for a in [0,1,0,1,0,2]:
            s = task.ENV.step(s, a)
        out = step(s, learner, 0)
        assert bool(out.done) and float(out.reward) == 1.
        ended = step(out.state, learner, 0)
        assert float(ended.reward) == 0.
    assert seen == {0,1}


@pytest.fixture(scope='module')
def agent():
    from training.learners.connect_four.build import build_agent
    result = build_agent(config_for('cce_decay', 0))
    assert isinstance(result, Agent)
    return result


def test_eval_fixed_identity_rng_and_reload(agent, tmp_path):
    before = agent.rng_identity()
    result = agent.evaluate()
    assert result['unique_start_positions'] == 1000
    assert result['unique_game_paths'] > 2
    assert agent.rng_identity() == before
    p = tmp_path/'last.pkl'
    agent.save(p)
    with p.open('rb') as f:
        restored = pickle.load(f)
    assert agent.evaluator.evaluate(restored['params']) == result
    assert restored['eval_bank_hash'] == config_for('per', 4)['eval_bank_hash']


def test_all_foes_use_legal_actions():
    s = task.ENV.init(jax.random.PRNGKey(9))
    for _ in range(6):
        s = task.ENV.step(s, 0)
    assert not bool(task.legal_mask(s)[0])
    f = jax.jit(foe)
    for policy in range(5):
        a = int(f(s, policy, jax.random.PRNGKey(0)))
        assert bool(task.legal_mask(s)[a])


def test_score_before_draw_and_finite_update(agent):
    # Tiny collection fixture; the production recipe stays frozen.
    cfg = agent.config.copy()
    agent.config.update(n_envs=2, collect_steps=2, batch_size=4)
    agent.carry = task.initial_carry(jax.random.split(jax.random.PRNGKey(50),2))
    try:
        agent.collect()
        assert (agent.buffer.scored_at[:len(agent.buffer)] >= 0).all()
        batch, indices, weights = agent.sample()
        params, opt, td, diag = agent.update_fn(agent.params, agent.target_params, agent.opt_state,batch,weights)
        assert np.isfinite(np.asarray(diag)).all()
        assert np.isfinite(np.asarray(td)).all()
        assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(params),jax.tree.leaves(agent.params)))
        agent.buffer.scored_at[0] = -1
        with pytest.raises(AssertionError, match='Unscored'):
            agent.sample()
        assert agent.total_draws == 4
    finally:
        agent.config = cfg
