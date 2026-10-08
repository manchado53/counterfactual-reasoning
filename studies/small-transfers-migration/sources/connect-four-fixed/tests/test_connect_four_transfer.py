"""Real CPU semantics and learning tests; PettingZoo is an isolated test dependency."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from counterfactual_rl.envs import connect_four_transfer as env
from counterfactual_rl.agents.connect_four.transfer import (
    Trainer, Network, ARMS, build_rollouts, consequence, targets,
)
from counterfactual_rl.agents.connect_four.transfer_run import checked_eval, source_hashes, effective_config


def tree_equal(a, b):
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(left, right)


def board_state(board, player=0):
    s = env.ENV.init(jax.random.PRNGKey(0))
    x = s._x._replace(board=jnp.asarray(board, jnp.int32).reshape(42), color=jnp.int32(player))
    s = s.replace(_x=x, current_player=jnp.int32(player),
                  legal_action_mask=jnp.asarray(np.asarray(board)[0] == -1))
    return s.replace(observation=env.ENV.observe(s, s.current_player))


@pytest.mark.parametrize('direction', [(0, 1), (1, 0), (1, 1), (1, -1)])
def test_wins_all_directions_against_pettingzoo(direction):
    from pettingzoo.classic.connect_four.connect_four import raw_env
    dr, dc = direction
    board = np.full((6, 7), -1, np.int32)
    r, c = (5, 0) if dr == 0 else (2, 3 if dc == -1 else 0)
    cells = [(r + i * dr, c + i * dc) for i in range(4)]
    # Support each target square. Test the rule kernel, not opening reachability.
    for rr, cc in cells:
        board[rr + 1:, cc] = 1
    for rr, cc in cells[:3]:
        board[rr, cc] = 0
    action = cells[-1][1]
    pz = raw_env()
    pz.reset()
    pz.board = (board + 1).reshape(-1).tolist()
    pz.step(action)
    s = env.ENV.step(board_state(board), jnp.int32(action))
    assert bool(s.terminated)
    assert float(s.rewards[0]) == pz.rewards['player_0'] == 1.
    np.testing.assert_array_equal(np.asarray(s._x.board) + 1, pz.board)


def test_legal_games_match_pettingzoo_and_draw():
    from pettingzoo.classic.connect_four.connect_four import raw_env
    # A reachable draw found by a fixed search, then independently checked here.
    rng = np.random.default_rng(123)
    saw_draw = False
    step = jax.jit(env.ENV.step)
    for game in range(250):
        pz = raw_env()
        pz.reset(seed=game)
        s = env.ENV.init(jax.random.PRNGKey(game))
        first_id = int(s.current_player)
        for ply in range(42):
            player_name = pz.agent_selection
            pz_obs = pz.observe(player_name)
            np.testing.assert_array_equal(s.observation, pz_obs['observation'])
            np.testing.assert_array_equal(env.legal_mask(s), pz_obs['action_mask'])
            action = int(rng.choice(np.flatnonzero(pz_obs['action_mask'])))
            s = step(s, jnp.int32(action))
            pz.step(action)
            np.testing.assert_array_equal(np.asarray(s._x.board) + 1, pz.board)
            assert bool(s.terminated) == all(pz.terminations.values())
            if s.terminated:
                assert float(s.rewards[first_id]) == pz.rewards['player_0']
                assert float(s.rewards[1 - first_id]) == pz.rewards['player_1']
                saw_draw |= not np.asarray(s.rewards).any()
                assert not np.asarray(env.legal_mask(s)).any()
                break
        if saw_draw and game >= 10:
            break
    assert saw_draw


def test_opponent_ties_full_columns_and_win_before_block():
    board = np.full((6, 7), -1, np.int32)
    assert int(env.opponent(board_state(board))) == 3
    board[:, 3] = [0, 1, 0, 1, 0, 1]
    assert int(env.opponent(board_state(board))) == 2
    board[:] = -1
    board[3:, 0] = 0
    board[3:, 6] = 1
    assert int(env.opponent(board_state(board))) == 0  # win beats block
    board[3:, 0] = -1
    assert int(env.opponent(board_state(board))) == 6  # block beats center


def test_paired_terminal_reply_and_absorption():
    board = np.full((6, 7), -1, np.int32)
    board[3:, 0] = 1
    s = board_state(board)
    out = jax.jit(env.step)(s, jnp.int32(0), jnp.int32(3))
    assert out.done and out.reward == -1 and out.board_moves == 2
    again = env.step(out.state, jnp.int32(0), jnp.int32(0))
    assert again.reward == 0 and again.board_moves == 0
    tree_equal(out.state, again.state)
    board[3:, 0] = 0
    out = env.step(board_state(board), jnp.int32(0), jnp.int32(0))
    assert out.done and out.reward == 1 and out.board_moves == 1


def test_roles_opening_and_split_carry():
    net = Network()
    params = net.init(jax.random.PRNGKey(1), jnp.zeros(84))
    carry = env.initial_carry(jax.random.split(jax.random.PRNGKey(2), 4))
    np.testing.assert_array_equal(carry.role, [0, 1, 0, 1])
    np.testing.assert_array_equal(carry.board_length, [0, 1, 0, 1])
    np.testing.assert_array_equal(carry.state.current_player, carry.learner)
    collect = env.build_collect(net)
    keys = jax.random.split(jax.random.PRNGKey(3), 64 * 4).reshape(64, 4, 2)
    whole, rows = collect(params, jnp.float32(1.), carry, keys)
    part, first = collect(params, jnp.float32(1.), carry, keys[:13])
    part, second = collect(params, jnp.float32(1.), part, keys[13:])
    tree_equal(whole, part)
    joined = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), first, second)
    tree_equal(rows, joined)
    assert np.asarray(rows['done']).any()
    assert not np.asarray(rows['state'].terminated).any()
    assert np.asarray(rows['episode_length']).max() <= 21
    done = np.asarray(rows['done'])
    np.testing.assert_array_equal(np.asarray(rows['board_length'])[done],
        2 * np.asarray(rows['episode_length'])[done] + np.asarray(rows['role'])[done]
        - (np.asarray(rows['r'])[done] == 1))
    ids = np.unique(np.asarray(rows['episode_id']))
    np.testing.assert_array_equal(ids, np.arange(int(whole.next_id)))


def test_masked_targets_weights_and_single_action():
    online = jnp.array([[900., 3., 4.], [9., 8., 7.]])
    target = jnp.array([[100., 20., 30.], [100., 20., 30.]])
    mask = jnp.array([[False, True, True], [False, False, False]])
    np.testing.assert_allclose(targets(jnp.array([0., -1.]), jnp.array([False, True]),
                                     online, target, mask), [29.7, -1.])
    ret = jnp.array([[1., -1., .5, 0., 0., 0., 0.]])
    q = jnp.array([[1., 0., 1000., 0., 0., 0., 0.]])
    m = jnp.array([[True, True, False, False, False, False, False]])
    np.testing.assert_allclose(consequence(ret, q, m, jnp.array([0])), [2.])
    np.testing.assert_array_equal(consequence(ret, q, m.at[0, 1].set(False), jnp.array([0])), [0.])


def test_horizon60_no_bootstrap_and_deterministic_n1():
    t = Trainer('uniform', 0, n_envs=2, capacity=128)
    states, players = t.carry.state, t.carry.learner
    keys = jax.random.split(jax.random.PRNGKey(9), 14).reshape(2, 7, 2)
    returns, done = t.rollout_fn(t.params, t.target_params, states, players, jnp.arange(7), keys)
    assert np.asarray(done).all()
    huge_targets = jax.tree.map(lambda a: jnp.ones_like(a) * 100., t.target_params)
    again, _ = t.rollout_fn(t.params, huge_targets, states, players, jnp.arange(7), keys + 1)
    np.testing.assert_array_equal(returns, again)
    assert np.isfinite(returns).all() and np.abs(returns).max() <= 1.


@pytest.mark.parametrize('arm', ARMS)
def test_real_updates_reload_rng_and_coverage(arm, tmp_path):
    np.random.seed(5)
    t = Trainer(arm, 0, n_envs=4, capacity=256)
    initial = copy.deepcopy(t.params)
    t.learn_block(32)
    assert t.steps == 128 and t.updates == t.attempts == 8
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(t.params)))
    assert np.isfinite(list(t.last_diag.values())).all()
    if t.cce:
        assert np.all(t.buffer.scored_at[:len(t.buffer)] == 0)
    metrics = checked_eval(t)
    assert metrics['unique_role_cases'] == 2 and not metrics['independent_trials']
    path = tmp_path / 'checkpoint.pkl'
    t.save(path)
    restored = Trainer.load(path)
    assert checked_eval(restored) == metrics
    assert restored.resumable is False
    tree_equal(t.params, restored.params)
    tree_equal(t.opt_state, restored.opt_state)
    assert source_hashes()


def test_score_matches_scipy_scalar_with_underflow_and_masks():
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(9)
    returns = rng.uniform(-1., 1., (128, 7)).astype(np.float32)
    q = rng.normal(size=(128, 7)).astype(np.float32)
    masks = rng.uniform(size=(128, 7)) > .3
    masks[:, 0] = True
    taken = np.zeros(128, np.int32)
    q[0] = [1000., 4., 0., 2., 0., 0., 0.]  # all alternative weights underflow
    masks[1] = [True, False, False, False, False, False, False]
    p = np.asarray(jax.nn.softmax(jnp.where(masks, jnp.asarray(q) / .5, -jnp.inf), -1))
    expected = []
    for i in range(128):
        alternatives = [a for a in range(7) if masks[i, a] and a != taken[i]]
        ds = [wasserstein_distance(returns[i, taken[i]:taken[i]+1], returns[i, a:a+1]) for a in alternatives]
        weights = [float(p[i, a]) for a in alternatives]
        expected.append(sum(w * d for w, d in zip(weights, ds)) / sum(weights)
                        if sum(weights) > 0 else (float(np.mean(ds)) if ds else 0.))
    actual = consequence(returns, jnp.asarray(q), jnp.asarray(masks), jnp.asarray(taken))
    np.testing.assert_array_equal(actual, expected)


def test_effective_config_marks_baseline_cce_inactive():
    for arm in ARMS:
        cfg = effective_config(arm, 0)
        assert cfg['cce_enabled'] == arm.startswith('cce')
        assert cfg['score_on_insert'] == arm.startswith('cce')
        assert cfg['normalise_priorities'] == ('rank' if arm.startswith('cce') else 'none')


def test_target_copy_phase_after_first_update():
    t = Trainer('uniform', 0, n_envs=4, capacity=128)
    original = t.update_fn
    used_targets, results = [], []
    def record(params, target, *args):
        used_targets.append(jax.device_get(target))
        result = original(params, target, *args)
        results.append(jax.device_get(result[0]))
        return result
    t.update_fn = record
    t.learn_block(32)
    tree_equal(used_targets[1], results[0])
    tree_equal(used_targets[-1], results[0])
    tree_equal(t.target_params, results[0])
